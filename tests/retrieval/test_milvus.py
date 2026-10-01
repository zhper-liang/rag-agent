"""仅显式启用时建立独立 l2_ 集合；HTTP 桩 + 真实 Milvus 端到端验证。

L2_MILVUS=1 uv run --env-file .env pytest -s tests/retrieval/test_milvus.py
不启动容器，不使用 GPU，不访问或释放其他线的集合。
"""

import asyncio
import os
import time
import uuid

import httpx
import pytest
from pymilvus import MilvusClient

from ragagent.context.build import build_context
from ragagent.graph.deps import Deps
from ragagent.index.schema import create_collection
from ragagent.retrieval.expand import expand
from ragagent.retrieval.hybrid import retrieve
from ragagent.retrieval.rerank import rerank
from tests.context.helpers import chunk, config, link
from tests.stubs.embed_server import create_app, vector


@pytest.mark.skipif(os.environ.get("L2_MILVUS") != "1", reason="显式启用独立 Milvus 实跑")
def test_real_milvus_hybrid_rerank_expand_context():
    coll = "l2_pipeline_" + uuid.uuid4().hex[:12]
    cfg = config(**{"services.milvus.collection": coll, "retrieval.timeout_ms": 15000})
    m = cfg.services.milvus
    client = MilvusClient(uri=m.uri, user=os.environ.get(m.user_env, ""),
                          password=os.environ.get(m.password_env, ""))
    created = False
    try:
        assert not client.has_collection(coll)
        create_collection(client, coll, cfg.index.dim)
        created = True
        query = "错误码 E00042 的解决方法"
        rows = []
        for i in range(1000):
            text = query + "：检查连接配置并重试。" if i == 42 else f"错误码 E{i:05d}：测试说明 {i}"
            row = chunk(i % 50, doc_id=f"{i // 50:016x}", text=text,
                        tokens=20, section_idx=i % 50 // 5, section_tokens=100)
            row.pop("score")
            row.pop("rerank_score")
            row["dense"] = vector(query if i == 42 else text, cfg.index.dim)
            rows.append(row)
        for start in range(0, len(rows), 50):
            link(*rows[start:start + 50])
        for start in range(0, len(rows), 200):
            client.insert(coll, rows[start:start + 200])
        client.flush(coll)
        deadline = time.monotonic() + 600
        indexes = ("dense", "sparse", "doc_id", "type", "file_type", "searchable")
        while True:
            states = {field: client.describe_index(coll, field) for field in indexes}
            if all(s.get("indexed_rows") == s.get("total_rows") == 1000 for s in states.values()):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("L2 索引构建超时")
            time.sleep(2)
        print("L2_MILVUS_INDEX", {f: (s["index_type"], s["indexed_rows"]) for f, s in states.items()})
        client.load_collection(coll)

        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(cfg.index.dim)),
                                         base_url="http://l2-stub") as stub:
                deps = Deps(cfg=cfg, count=len, milvus=client, embed_client=stub, rerank_client=stub)
                candidates, retrieval = await retrieve([query], None, deps)
                assert candidates and candidates[0].chunk_id == rows[42]["chunk_id"]
                hits, result = await rerank(query, candidates, deps, retrieval)
                expansions = await asyncio.to_thread(expand, client, coll, hits, cfg)
                blocks, images = build_context(hits, expansions, cfg)
                assert hits and result.top_score > 0 and not result.degraded
                assert any(expansions.values()) and blocks
                assert rows[42]["chunk_id"] in {cid for b in blocks for cid in b.chunk_ids}
                print("L2_PIPELINE_OK", {"collection": coll, "candidates": len(candidates),
                      "top_score": result.top_score, "hits": [h.chunk_id for h in hits],
                      "expansions": sum(map(len, expansions.values())), "context_blocks": len(blocks),
                      "images": len(images), "timings_ms": result.timings_ms})
        asyncio.run(run())
    finally:
        if created:
            client.drop_collection(coll)
            print("L2_CLEANED_OWN_COLLECTION", coll)
        client.close()
