"""Gold quote 对齐到当前索引；不把切分 ID 写入评测集。"""
from collections import defaultdict


def align(quote: str, chunks: list[dict]) -> set[str] | None:
    """按 seq 排序的同文档当前版本 text/code chunks；去掉拆分重叠。"""
    if not quote:
        return None
    parts, spans, offset = [], [], 0
    for c in chunks:
        overlap = c.get("meta", {}).get("overlap_chars", 0)
        if not isinstance(overlap, int) or not 0 <= overlap <= len(c["text"]):
            raise ValueError("invalid meta.overlap_chars")
        body = c["text"][overlap:]
        spans.append((offset, offset + len(body), c["chunk_id"]))
        offset += len(body)
        parts.append(body)
    start = "".join(parts).find(quote)
    if start < 0:
        return None
    end = start + len(quote)
    return {cid for a, b, cid in spans if a < end and b > start}


def align_gold(gold: list[dict], chunks: list[dict]) -> list[set[str]] | None:
    """每个 gold 片段对应一组可命中的 ID；任一片段 stale 则整题排除。

    gold.version 是生成时版本；使用当前版本找 quote，而非按旧版本过滤。
    图片不与 text/code 拼接，只在独立图片正文中做逐字匹配。
    """
    docs = defaultdict(list)
    for c in chunks:
        docs[c["doc_id"]].append(c)
    result = []
    for g in gold:
        rows = docs.get(g["doc_id"], [])
        latest = max((c["version"] for c in rows), default=None)
        rows = sorted((c for c in rows if c["version"] == latest), key=lambda c: c["seq"])
        ids = align(g["quote"], [c for c in rows if c["type"] in {"text", "code"}])
        if ids is None:
            ids = {c["chunk_id"] for c in rows if c["type"] == "image" and g["quote"] in c["text"]}
        if not ids:
            return None
        result.append(ids)
    return result
