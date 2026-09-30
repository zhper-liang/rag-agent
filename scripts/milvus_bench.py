"""回答 向量化建库.md §12 的两个实测问题，结论写在 §6.4。

    uv run --env-file .env python scripts/milvus_bench.py mmap     # SPARSE_INVERTED_INDEX 能否 mmap
    uv run --env-file .env python scripts/milvus_bench.py recall   # PQ 比例下 DISKANN 的 Recall@k

两个测试各用一个独立集合（bench_sparse / bench_dense），不碰正式集合；开始前会 release 其他已加载集合，
让内存读数干净，结束后不会自动重新加载。数据是合成的：
- mmap：Zipf 分布的中文假词 + 标识符，只关心倒排表规模；
- recall：分层聚类 + 低内在维度的 1024 维向量，和真实 bge-m3 向量分布不同，真实数据入库后要按 §10 再测一次。
内存读的是容器里 milvus 进程的 /proc/<pid>/status 与 smaps（milvus 在容器内不是 PID 1）。
"""

import argparse
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml
from pymilvus import DataType, Function, FunctionType, MilvusClient

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8"))
CONTAINER = "milvus-standalone"
MMAP_DIR = "/milvus_local/data/mmap"      # milvus.yaml localStorage.path + mmap/


def connect() -> MilvusClient:
    m = CFG["services"]["milvus"]
    return MilvusClient(uri=m["uri"], user=os.environ.get(m["user_env"], ""),
                        password=os.environ.get(m["password_env"], ""))


def sh(cmd: str) -> str:
    return subprocess.run(["docker", "exec", CONTAINER, "sh", "-c", cmd],
                          check=True, capture_output=True, text=True).stdout


def milvus_pid() -> str:
    for line in sh("ps -eo pid,args").splitlines():
        pid, _, args = line.strip().partition(" ")
        if args.startswith("milvus run"):      # PID 1 是 tini -- milvus run ...，要跳过
            return pid
    raise RuntimeError("容器里找不到 milvus 进程")


def mem_mb() -> dict[str, float]:
    """RssAnon / RssFile，以及映射自 mmap 目录的常驻页（smaps 按文件汇总）。"""
    pid = milvus_pid()
    st = dict(l.split(":", 1) for l in sh(f"cat /proc/{pid}/status").splitlines() if ":" in l)
    mapped = sh(f"awk '/^[0-9a-f]+-[0-9a-f]+ /{{f=$6}} /^Rss:/{{if (index(f, \"{MMAP_DIR}\")==1) s+=$2}} "
                f"END{{print s+0}}' /proc/{pid}/smaps")
    kb = lambda k: int(st[k].split()[0])
    return {"anon": kb("RssAnon") / 1024, "file": kb("RssFile") / 1024, "mmap_dir": int(mapped) / 1024}


def release_all(client: MilvusClient) -> None:
    for name in client.list_collections():
        client.release_collection(name)


def wait_indexed(client: MilvusClient, coll: str, field: str, timeout_s: float = 3600) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        d = client.describe_index(coll, field)
        if d.get("indexed_rows") == d.get("total_rows") > 0:
            print(f"index {field}: {d['index_type']} {d['indexed_rows']}/{d['total_rows']}")
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"{field} 索引未建完：{d.get('indexed_rows')}/{d.get('total_rows')}")
        time.sleep(5)


# ---------------------------------------------------------------- mmap

SPARSE = "bench_sparse"


def fake_texts(n: int, seed: int = 0) -> list[str]:
    """Zipf 分布的假中文词（2~4 字）+ 少量标识符，每段约 150 词。"""
    rng = random.Random(seed)
    chars = [chr(c) for c in range(0x4E00, 0x4E00 + 3000)]
    vocab = ["".join(rng.choices(chars, k=rng.choice((2, 2, 3, 4)))) for _ in range(150_000)]
    vocab += [f"{rng.choice(('get', 'set', 'max', 'num'))}_{rng.choice(('user', 'token', 'size', 'id'))}{i}"
              for i in range(50_000)]
    rng.shuffle(vocab)
    ranks = np.random.default_rng(seed).zipf(1.1, size=(n, 150))
    ranks = (ranks - 1) % len(vocab)   # 长尾折回词表，避免全部压在最后一个词上
    return ["，".join(" ".join(vocab[r] for r in row[i:i + 8]) for i in range(0, 150, 8)) for row in ranks]


def build_sparse(client: MilvusClient, rows: int) -> None:
    if client.has_collection(SPARSE) and client.get_collection_stats(SPARSE)["row_count"] == rows:
        return
    if client.has_collection(SPARSE):
        client.drop_collection(SPARSE)
    s = client.create_schema(auto_id=False)
    s.add_field("id", DataType.INT64, is_primary=True)
    s.add_field("embed_text", DataType.VARCHAR, max_length=65535,
                enable_analyzer=True, analyzer_params={"type": "chinese"})
    s.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    s.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                            input_field_names=["embed_text"], output_field_names=["sparse"]))
    idx = client.prepare_index_params()
    idx.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client.create_collection(SPARSE, schema=s, index_params=idx)
    texts = fake_texts(rows)
    for start in range(0, rows, 2000):
        client.insert(SPARSE, [{"id": i, "embed_text": texts[i]} for i in range(start, min(start + 2000, rows))])
    client.flush(SPARSE)
    print("inserted", client.get_collection_stats(SPARSE))


def settle(client: MilvusClient, coll: str, field: str) -> None:
    """插入后后台还会 compaction，segment 被替换会让加载超时；先手动合并完、索引建完再测。"""
    wait_indexed(client, coll, field)
    job = client.compact(coll)
    while client.get_compaction_state(job) != "Completed":
        time.sleep(5)
    wait_indexed(client, coll, field)


def bench_mmap(client: MilvusClient, rows: int) -> None:
    release_all(client)
    build_sparse(client, rows)
    settle(client, SPARSE, "sparse")
    queries = [t[:40] for t in fake_texts(20, seed=1)]
    for mmap in (False, True):
        client.release_collection(SPARSE)          # 带 index_params 建表会自动加载；改索引属性要先释放
        client.alter_index_properties(SPARSE, "sparse", {"mmap.enabled": mmap})
        time.sleep(3)
        before, t0 = mem_mb(), time.monotonic()
        client.load_collection(SPARSE, timeout=3600)
        load_s = time.monotonic() - t0
        client.search(SPARSE, data=queries, anns_field="sparse", limit=10, search_params={"metric_type": "BM25"})
        after = mem_mb()
        d = {k: round(after[k] - before[k], 1) for k in after}
        print(f"sparse 索引 mmap.enabled={client.describe_index(SPARSE, 'sparse').get('mmap.enabled')}: "
              f"加载 {load_s:.0f} s  ΔRssAnon={d['anon']} MB  ΔRssFile={d['file']} MB  "
              f"mmap 目录常驻={after['mmap_dir']:.1f} MB  mmap 目录占用 {sh(f'du -sm {MMAP_DIR}').split()[0]} MB",
              flush=True)
        client.release_collection(SPARSE)
        time.sleep(10)
    client.alter_index_properties(SPARSE, "sparse", {"mmap.enabled": True})   # 恢复成与 milvus.yaml 一致


# ---------------------------------------------------------------- recall

DENSE = "bench_dense"      # 实际集合名带数据类型后缀：bench_dense_clustered / bench_dense_gaussian


def fake_vectors(n: int, dim: int, seed: int, data: str) -> np.ndarray:
    """clustered：两层聚类放在 64 维潜空间里，再随机正交投影到 dim 维并加噪声（偏容易）；
    gaussian：各向同性高斯，没有任何结构，是 ANN 召回的下界。最后都归一化。"""
    if data == "gaussian":
        x = np.random.default_rng(seed).normal(size=(n, dim))
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    rng = np.random.default_rng(0)                   # 簇中心和投影固定，base 与 query 同分布
    top = rng.normal(size=(200, 64))
    sub = top[rng.integers(0, 200, 5000)] * 0.7 + rng.normal(size=(5000, 64)) * 0.5
    proj = np.linalg.qr(rng.normal(size=(dim, 64)))[0].T
    rng = np.random.default_rng(seed)
    lat = sub[rng.integers(0, 5000, n)] + rng.normal(size=(n, 64)) * 0.35
    x = lat @ proj + rng.normal(size=(n, dim)) * 0.02
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def build_dense(client: MilvusClient, coll: str, base: np.ndarray) -> None:
    if client.has_collection(coll) and client.get_collection_stats(coll)["row_count"] == len(base):
        return
    if client.has_collection(coll):
        client.drop_collection(coll)
    s = client.create_schema(auto_id=False)
    s.add_field("id", DataType.INT64, is_primary=True)
    s.add_field("dense", DataType.FLOAT_VECTOR, dim=base.shape[1])
    idx = client.prepare_index_params()
    idx.add_index("dense", index_type="DISKANN", metric_type="COSINE")
    client.create_collection(coll, schema=s, index_params=idx)
    for start in range(0, len(base), 2000):
        client.insert(coll, [{"id": i, "dense": base[i].tolist()} for i in range(start, min(start + 2000, len(base)))])
    client.flush(coll)
    print("inserted", client.get_collection_stats(coll))


def bench_recall(client: MilvusClient, rows: int, queries: int, data: str) -> None:
    dim, coll = CFG["index"]["dim"], f"{DENSE}_{data}"
    base, q = fake_vectors(rows, dim, 1, data), fake_vectors(queries, dim, 2, data)
    ks = (10, CFG["retrieval"]["dense_top_k"])
    sims = q @ base.T                                  # 暴力检索（等价于 FLAT）当标准答案
    gt = {k: np.argsort(-sims, axis=1)[:, :k] for k in ks}
    del sims
    release_all(client)
    build_dense(client, coll, base)
    settle(client, coll, "dense")
    client.load_collection(coll, timeout=3600)
    print(f"PQCodeBudgetGBRatio={sh('grep PQCodeBudgetGBRatio /milvus/configs/milvus.yaml').split(':')[1].split('#')[0].strip()} "
          f"data={data} rows={rows} dim={dim} indexed={client.describe_index(coll, 'dense')['indexed_rows']}")
    for sl in (16, 32, 50, 64, 100, 200):
        line = []
        for k in ks:
            if sl < k:
                continue
            hit = 0
            for start in range(0, queries, 100):
                res = client.search(coll, data=q[start:start + 100].tolist(), anns_field="dense", limit=k,
                                    search_params={"metric_type": "COSINE", "params": {"search_list": sl}})
                hit += sum(len({h["id"] for h in r} & set(gt[k][start + j].tolist())) for j, r in enumerate(res))
            line.append(f"Recall@{k}={hit / (queries * k):.4f}")
        print(f"search_list={sl}: " + "  ".join(line))
    client.release_collection(coll)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("mmap", "recall"))
    ap.add_argument("--rows", type=int, default=None, help="mmap 默认 60000，recall 默认 200000")
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--data", choices=("clustered", "gaussian"), default="clustered", help="只对 recall 有效")
    a = ap.parse_args()
    client = connect()
    if a.what == "mmap":
        bench_mmap(client, a.rows or 60_000)
    else:
        bench_recall(client, a.rows or 200_000, a.queries, a.data)


if __name__ == "__main__":
    main()
