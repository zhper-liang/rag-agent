"""硬过滤永不放宽；任意冲突让整套软过滤作废（预处理 §8.3）。"""

from ragagent.retrieval.filters import validate_filters


def effective_filters(hard: dict | None, soft: dict | None) -> dict | None:
    hard_valid, soft_valid = validate_filters(hard), validate_filters(soft)
    if not soft_valid:
        return hard_valid or None
    if not hard_valid:
        return soft_valid
    merged = {}
    for key in ("file_types", "doc_ids"):
        h, s = hard_valid.get(key), soft_valid.get(key)
        if h and s:
            inter = sorted(set(h) & set(s))
            if not inter:
                return hard_valid
            merged[key] = inter
        elif h or s:
            merged[key] = h or s
    return merged or None
