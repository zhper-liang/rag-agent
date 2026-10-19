"""Milvus 检索过滤的唯一用户输入边界（检索与重排 §4.2）。"""

import re

DOC_ID_RE = re.compile(r"^[0-9a-f]{16}$")
FILE_TYPES = frozenset({"pdf", "doc", "docx", "ppt", "pptx", "png", "jpg", "jpeg", "md", "txt"})
FILTER_FIELDS = frozenset({"file_types", "doc_ids"})
MAX_FILTER_ITEMS = 100


def validate_filters(filters: dict | None) -> dict[str, list[str]]:
    """校验全部输入后才允许拼接；拒绝未知字段、非数组和非字符串元素。"""
    if filters is None:
        return {}
    if not isinstance(filters, dict) or not filters.keys() <= FILTER_FIELDS:
        raise ValueError("过滤条件只能包含 file_types 和 doc_ids")
    out = {}
    for key, values in filters.items():
        if not isinstance(values, list) or len(values) > MAX_FILTER_ITEMS:
            raise ValueError("过滤值必须为最多 100 项的数组")
        for value in values:
            if not isinstance(value, str):
                raise ValueError("过滤值必须为字符串")
            if key == "doc_ids" and not DOC_ID_RE.fullmatch(value):
                raise ValueError("非法 doc_id")
            if key == "file_types" and value not in FILE_TYPES:
                raise ValueError("不支持的文件类型")
        if values:
            out[key] = sorted(set(values))
    return out


def build_expr(filters: dict | None, *, time_doc_ids: list[str] | None = None) -> str:
    valid = validate_filters(filters)
    parts = ["searchable == true"]
    for key, field in (("file_types", "file_type"), ("doc_ids", "doc_id")):
        if values := valid.get(key):
            parts.append(field + " in [" + ",".join(f'"{v}"' for v in values) + "]")
    if time_doc_ids is not None:
        if not isinstance(time_doc_ids, list) or not time_doc_ids or not all(
                isinstance(i, str) and DOC_ID_RE.fullmatch(i) for i in time_doc_ids):
            raise ValueError("非法内部时间范围 doc_id")
        parts.append('doc_id in [' + ','.join(f'"{i}"' for i in sorted(set(time_doc_ids))) + ']')
    return " and ".join(parts)
