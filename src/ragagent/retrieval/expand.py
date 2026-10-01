"""批量取回扩展：id 一次、整节一次；角色归属与全局图片限额。"""

from collections import defaultdict

from ragagent.ingest.chunking.models import parse_chunk_id
from ragagent.retrieval.filters import DOC_ID_RE
from ragagent.retrieval.merge import CHUNK_FIELDS, chunk_id_expr
from ragagent.retrieval.models import Expansion, Hit


def _section_key(c):
    doc, version, idx = c["doc_id"], c["version"], c["section_idx"]
    if not isinstance(doc, str) or not DOC_ID_RE.fullmatch(doc) or type(version) is not int or type(idx) is not int:
        raise ValueError("非法整节条件")
    return doc, version, idx


def section_expr(keys) -> str:
    parts = []
    for doc, version, idx in sorted(set(keys)):
        _section_key({"doc_id": doc, "version": version, "section_idx": idx})
        parts.append(f'(doc_id == "{doc}" and version == {version} and section_idx == {idx})')
    if not parts:
        raise ValueError("整节条件不能为空")
    return "(" + " or ".join(parts) + ') and type != "image"'


def _split_sibling(hit, other):
    if _section_key(hit) != _section_key(other) or not other["meta"].get("split_part"):
        return False
    try:
        index, total = map(int, hit["meta"]["split_part"].split("/"))
        sibling, sibling_total = map(int, other["meta"]["split_part"].split("/"))
        return total == sibling_total and abs(index - sibling) == 1
    except (KeyError, ValueError, AttributeError):
        return False


def expand(client, coll: str, hits, cfg) -> dict[str, list[Expansion]]:
    hits = [Hit.model_validate(h).model_dump() for h in hits]
    all_hit_ids = {h["chunk_id"] for h in hits}
    requests, sections = defaultdict(list), defaultdict(list)
    image_ids = set()
    for h in hits:
        hid = h["chunk_id"]
        if h["type"] in {"text", "code"}:
            small = h["section_tokens"] <= cfg.expansion.section_max_tokens and h["meta"].get("strategy") != "section"
            if small:
                sections[_section_key(h)].append(hid)
            for cid in (h["prev_id"], h["next_id"]):
                if cid:
                    # 小节也取邻块，为整节放不下或数据库 limit 截断提供回退。
                    requests[cid].append((hid, "neighbor"))
                    if h["meta"].get("split_part"):
                        requests[cid].append((hid, "split_sibling"))
            if h["type"] == "code":
                for cid in h["related_ids"]:
                    requests[cid].append((hid, "lead_in"))
            count = 0
            for cid in h["image_ids"]:
                if count >= cfg.expansion.images_per_chunk or (cid not in image_ids and len(image_ids) >= cfg.expansion.images_total):
                    break
                if cid not in all_hit_ids:
                    requests[cid].append((hid, "attachment"))
                    image_ids.add(cid)
                    count += 1
        elif h["type"] == "image" and (anchor := h["meta"].get("anchor_id")):
            requests[anchor].append((hid, "anchor"))
    for cid in requests:
        parse_chunk_id(cid)  # 内部 id 也先校验，不能把损坏的库字段拼进去。
    wanted = sorted(set(requests) - all_hit_ids)
    rows = client.query(collection_name=coll, filter=chunk_id_expr(wanted), output_fields=CHUNK_FIELDS,
                        limit=len(wanted), timeout=cfg.retrieval.timeout_ms / 1000) if wanted else []
    out = {h["chunk_id"]: [] for h in hits}
    by_hit = {h["chunk_id"]: h for h in hits}
    seen = set()

    def add(hid, row, role):
        key = (hid, row["chunk_id"], role)
        if row["chunk_id"] in all_hit_ids or key in seen:
            return
        if role == "attachment" and row["type"] != "image":
            return
        if role != "attachment" and row["type"] == "image":
            return
        if role == "split_sibling" and not _split_sibling(by_hit[hid], row):
            return
        if (row["doc_id"], row["version"]) != (by_hit[hid]["doc_id"], by_hit[hid]["version"]):
            return  # 链和附件不应穿越文档/版本。
        seen.add(key)
        out[hid].append(Expansion(**row, role=role))

    for row in rows:
        for hid, role in requests[row["chunk_id"]]:
            add(hid, row, role)
    if sections:
        section_rows = client.query(collection_name=coll, filter=section_expr(sections), output_fields=CHUNK_FIELDS,
                                    limit=200, timeout=cfg.retrieval.timeout_ms / 1000)
        for row in sorted(section_rows, key=lambda c: (c["doc_id"], c["version"], c["seq"])):
            for hid in sections.get(_section_key(row), []):
                add(hid, row, "section")
    return out
