"""仅本线的协议 fake；真实 Milvus 验证另有 test_milvus.py。"""
import json
import re
import threading
import time


class FakeMilvus:
    def __init__(self, rows, *, failures=None, delay=0):
        self.rows = rows
        self.failures = failures or {}
        self.delay = delay
        self.calls = []
        self.active = self.max_active = 0
        self.lock = threading.Lock()

    def hybrid_search(self, **kwargs):
        self.calls.append(("hybrid", kwargs))
        query = kwargs["reqs"][1].data[0]
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            if query in self.failures:
                raise RuntimeError("hybrid unavailable")
            return [[{"entity": {k: r[k] for k in kwargs["output_fields"]}, "distance": 1 - i / 100}
                     for i, r in enumerate(self.rows) if r["searchable"]]]
        finally:
            with self.lock:
                self.active -= 1

    def search(self, **kwargs):
        self.calls.append(("dense", kwargs))
        if any(v == "both" for v in self.failures.values()):
            raise RuntimeError("dense unavailable")
        return [[{"entity": {k: r[k] for k in kwargs["output_fields"]}, "distance": .8 - i / 100}
                 for i, r in enumerate(self.rows) if r["searchable"]]]

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        expr = kwargs["filter"]
        if expr.startswith("chunk_id in "):
            ids = set(json.loads(expr[len("chunk_id in "):]))
            found = [r for r in self.rows if r["chunk_id"] in ids]
        else:
            keys = {(d, int(v), int(s)) for d, v, s in re.findall(
                r'doc_id == "([0-9a-f]{16})" and version == (\d+) and section_idx == (\d+)', expr)}
            found = [r for r in self.rows if (r["doc_id"], r["version"], r["section_idx"]) in keys and r["type"] != "image"]
        return [{k: r[k] for k in kwargs["output_fields"]} for r in found[:kwargs["limit"]]]


def row(chunk):
    return {k: v for k, v in chunk.items() if k not in {"score", "rerank_score", "role"}}
