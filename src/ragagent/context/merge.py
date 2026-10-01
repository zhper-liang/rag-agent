"""按文档/原文顺序合并，保留来源、去掉真实拆分段的重叠。"""

from __future__ import annotations

from collections import defaultdict

from ragagent.context.images import image_ref
from ragagent.context.models import ContextBlock


def rank_score(chunk: dict) -> float:
    score = chunk.get("rerank_score")
    return float(chunk.get("score", 0.0) if score is None else score)


def _section(c: dict) -> tuple:
    return c["doc_id"], c["version"], c["section_idx"]


def chunk_groups(chosen: dict[str, dict], roles: dict[str, str]) -> list[list[dict]]:
    """用于实际合并和预算的同一分组规则，避免预算和输出不一致。"""
    parent = {cid: cid for cid in chosen}

    def root(cid):
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid

    def join(a, b):
        ca, cb = chosen[a], chosen[b]
        if (ca["doc_id"], ca["version"]) == (cb["doc_id"], cb["version"]):
            parent[root(b)] = root(a)

    whole_sections = {_section(c) for cid, c in chosen.items() if roles[cid] == "section"}
    sections = {}
    for cid, c in chosen.items():
        if c["type"] == "image":
            continue
        for adjacent in (c["prev_id"], c["next_id"]):
            if adjacent in chosen and chosen[adjacent]["type"] != "image":
                join(cid, adjacent)
        if _section(c) in whole_sections:
            if _section(c) in sections:
                join(cid, sections[_section(c)])
            sections[_section(c)] = cid
    # 图片不入链，只附到确实已选的锚点。
    for cid, c in chosen.items():
        if c["type"] == "image":
            anchor = c["meta"].get("anchor_id")
            if anchor in chosen and chosen[anchor]["type"] != "image":
                join(anchor, cid)
    groups = defaultdict(list)
    for cid, c in chosen.items():
        groups[root(cid)].append(c)
    return [sorted(g, key=lambda c: c["seq"]) for g in groups.values()]


def _is_split_continuation(previous: dict, current: dict) -> bool:
    if _section(previous) != _section(current):
        return False
    if not (previous["next_id"] == current["chunk_id"] or current["prev_id"] == previous["chunk_id"]):
        return False
    try:
        p, total = map(int, previous["meta"]["split_part"].split("/"))
        q, qtotal = map(int, current["meta"]["split_part"].split("/"))
        return q == p + 1 and total == qtotal
    except (KeyError, ValueError, AttributeError):
        return False


def merge_text(chunks: list[dict]) -> str:
    text, previous = "", None
    for c in chunks:
        if c["type"] == "image":
            continue
        part = c["text"]
        continuation = previous is not None and _is_split_continuation(previous, c)
        if continuation:
            overlap = c["meta"].get("overlap_chars", 0)
            if (type(overlap) is not int or overlap < 0 or overlap > len(part)
                    or (overlap and not previous["text"].endswith(part[:overlap]))):
                raise ValueError("拆分段 overlap_chars 与原文不一致")
            part = part[overlap:]
        if previous is not None and previous["section_path"] != c["section_path"]:
            heading = c["section_path"].split(" > ")[-1]
            text += "\n\n## " + heading + "\n\n"
        elif previous is not None and not continuation:
            text += "\n\n"
        text += part
        previous = c
    return text


def merge_blocks(chosen: dict[str, dict], roles: dict[str, str], send_image: set[str],
                 attachments=None) -> list[ContextBlock]:
    groups = chunk_groups(chosen, roles)
    doc_scores = defaultdict(float)
    for c in chosen.values():
        if roles[c["chunk_id"]] == "hit":
            doc_scores[c["doc_id"]] = max(doc_scores[c["doc_id"]], rank_score(c))
    groups.sort(key=lambda g: (-doc_scores[g[0]["doc_id"]], g[0]["doc_id"], g[0]["seq"]))
    blocks = []
    for cite, group in enumerate(groups, 1):
        body = [c for c in group if c["type"] != "image"]
        first = body[0] if body else group[0]
        images = [image_ref(c, c["chunk_id"] in send_image) for c in group if c["type"] == "image"]
        for ref in attachments or []:
            if any(ref.chunk_id in c["image_ids"] for c in body):
                images.append(ref)
        pages = [p for c in group for p in (c["page_start"], c["page_end"]) if p >= 0]
        hit_ids = [c["chunk_id"] for c in group if roles[c["chunk_id"]] == "hit"]
        text = merge_text(body)
        captions = [f'[图: {c["text"]}]' for c in group if c["type"] == "image"]
        text = "\n\n".join([t for t in (text, *captions) if t])
        blocks.append(ContextBlock(
            cite=cite, doc_id=first["doc_id"], title=first["title"], section_path=first["section_path"],
            page_start=min(pages, default=-1), page_end=max(pages, default=-1), text=text,
            chunk_ids=[c["chunk_id"] for c in group], hit_ids=hit_ids,
            score=max((rank_score(c) for c in group if c["chunk_id"] in hit_ids), default=0.0),
            images=images,
        ))
    return blocks
