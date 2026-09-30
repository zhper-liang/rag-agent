"""Milvus 建表冒烟（开发计划.md §3.5）：create_collection、写入 1000 条假数据、建索引、hybrid search。

用独立的集合 rag_chunks_smoke，每次重建；不会碰 services.milvus.collection 指向的正式集合。
地址、维度、检索参数取 config.yaml；用户名密码取环境变量（授权关闭时可以为空）。

    uv run --env-file .env python scripts/milvus_smoke.py
"""

import hashlib
import math
import os
import random
import time
from pathlib import Path

import yaml
from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker

from ragagent.index.models import ChunkRow
from ragagent.index.schema import create_collection
from ragagent.ingest.chunking.models import make_chunk_id

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8"))
COLLECTION = "rag_chunks_smoke"
ROWS = 1000
DOCS = 20


def connect() -> MilvusClient:
    m = CFG["services"]["milvus"]
    return MilvusClient(uri=m["uri"], user=os.environ.get(m["user_env"], ""),
                        password=os.environ.get(m["password_env"], ""))


def vector(i: int, dim: int) -> list[float]:
    rng = random.Random(hashlib.sha256(f"fake-{i}".encode()).digest())
    v = [rng.gauss(0, 1) for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def make_row(i: int, dim: int) -> dict:
    doc_id, seq = f"{i % DOCS:016x}", i // DOCS
    text = f"测试文档 {i % DOCS} 第 {seq} 段，包含错误码 E{i:05d}。"
    path = f"测试章节 > 第{seq % 10 + 1}节"
    cid = make_chunk_id(doc_id, 1, seq)
    return ChunkRow(
        chunk_id=cid, doc_id=doc_id, version=1, seq=seq, tokens=32, section_idx=seq % 10 + 1,
        section_tokens=320, type="text", text=text, embed_text=f"假文档 {i % DOCS} > {path}\n{text}",
        content_hash=hashlib.sha256(text.encode()).hexdigest(), title=f"假文档 {i % DOCS}",
        section_path=path, source_path=f"test/{i % DOCS}.md", file_type="md", page_start=-1, page_end=-1,
        image_path="", prev_id=make_chunk_id(doc_id, 1, seq - 1) if seq else "",
        next_id=make_chunk_id(doc_id, 1, seq + 1) if seq + 1 < ROWS // DOCS else "",
        image_ids=[], related_ids=[], searchable=True, meta={"fake": True}, dense=vector(i, dim),
    ).model_dump()


INDEXED = ("dense", "sparse", "doc_id", "type", "file_type", "searchable")


def wait_indexed(client: MilvusClient, coll: str, timeout_s: float = 600) -> None:
    """flush 后索引异步构建；全部建完再加载，否则检索走的是暴力搜索。"""
    deadline = time.monotonic() + timeout_s
    while True:
        ds = {f: client.describe_index(coll, f) for f in INDEXED}
        if all(d.get("indexed_rows") == d.get("total_rows") > 0 for d in ds.values()):
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"索引未建完：{ {f: (d.get('indexed_rows'), d.get('total_rows')) for f, d in ds.items()} }")
        time.sleep(2)
    for f, d in ds.items():
        print(f"index {f}: {d['index_type']} {d['indexed_rows']}/{d['total_rows']}")


def main() -> None:
    dim, r = CFG["index"]["dim"], CFG["retrieval"]
    client = connect()
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)
    create_collection(client, COLLECTION, dim)

    rows = [make_row(i, dim) for i in range(ROWS)]
    for start in range(0, ROWS, 200):
        client.insert(COLLECTION, rows[start:start + 200])
    client.flush(COLLECTION)
    print("inserted", client.get_collection_stats(COLLECTION))
    wait_indexed(client, COLLECTION)

    client.load_collection(COLLECTION)
    for _ in range(60):
        if "Loaded" in str(client.get_load_state(COLLECTION)["state"]):
            break
        time.sleep(1)

    target = rows[42]
    expr = "searchable == true"
    dense = AnnSearchRequest(data=[target["dense"]], anns_field="dense", limit=r["dense_top_k"], expr=expr,
                             param={"metric_type": "COSINE", "params": {"search_list": r["search_list"]}})
    sparse = AnnSearchRequest(data=["E00042"], anns_field="sparse", limit=r["sparse_top_k"], expr=expr,
                              param={"metric_type": "BM25"})
    res = client.hybrid_search(COLLECTION, reqs=[dense, sparse],
                               ranker=WeightedRanker(r["weights"]["dense"], r["weights"]["sparse"]),
                               limit=5, output_fields=["chunk_id", "doc_id", "seq"])[0]
    print("hybrid_hits", [(h["entity"]["chunk_id"], round(h["distance"], 4)) for h in res])
    assert res and res[0]["entity"]["chunk_id"] == target["chunk_id"]
    print("HYBRID_SEARCH_OK")


if __name__ == "__main__":
    main()
