"""多查询最高分合并、最新版本和内容去重、候选回表。"""

import json

from ragagent.index.models import ChunkFields
from ragagent.ingest.chunking.models import parse_chunk_id
from ragagent.retrieval.models import Hit

CHUNK_FIELDS = list(ChunkFields.model_fields)


def chunk_id_expr(ids: list[str]) -> str:
    if not ids:
        raise ValueError("不能查询空 chunk_id 列表")
    for cid in ids:
        parse_chunk_id(cid)
    return "chunk_id in " + json.dumps(sorted(set(ids)), ensure_ascii=False)


def merge(results: list[list[dict]], limit: int) -> list[dict]:
    if limit < 0:
        raise ValueError("limit 必须非负")
    best = {}
    for hits in results:
        for h in hits:
            cid = h["chunk_id"]
            if cid not in best or h["score"] > best[cid]["score"]:
                best[cid] = dict(h)
    latest = {}
    for h in best.values():
        latest[h["doc_id"]] = max(latest.get(h["doc_id"], 0), h["version"])
    valid = [h for h in best.values() if h["version"] == latest[h["doc_id"]]]
    return sorted(valid, key=lambda h: (-h["score"], h["chunk_id"]))[:limit]


def dedup_by_content(hits: list[dict]) -> list[dict]:
    best = {}
    for h in hits:
        key = h["content_hash"]
        if key not in best or h["score"] > best[key]["score"]:
            best[key] = h
    return sorted(best.values(), key=lambda h: (-h["score"], h["chunk_id"]))


def fetch_candidates(client, coll: str, hits: list[dict], cfg) -> list[Hit]:
    if not hits:
        return []
    rows = client.query(collection_name=coll, filter=chunk_id_expr([h["chunk_id"] for h in hits]),
                        output_fields=CHUNK_FIELDS, limit=len(hits), timeout=cfg.timeout_ms / 1000)
    by_id = {r["chunk_id"]: r for r in rows}
    full = [Hit(**by_id[h["chunk_id"]], score=h["score"]).model_dump()
            for h in hits if h["chunk_id"] in by_id]
    return [Hit.model_validate(h) for h in dedup_by_content(full)]
