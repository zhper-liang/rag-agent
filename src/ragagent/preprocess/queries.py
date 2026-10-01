"""NFKC/大小写/空白去重，去重后再截断（问题预处理 §7）。"""

import unicodedata

from ragagent.preprocess.schema import Preprocessed


def query_key(query: str) -> str:
    return "".join(unicodedata.normalize("NFKC", query).lower().split())


def dedup_queries(queries, *, limit=3, tried=()) -> list[str]:
    seen = {query_key(q) for q in tried}
    out = []
    for query in queries:
        query = unicodedata.normalize("NFKC", query).strip()
        key = query_key(query)
        if key and key not in seen:
            seen.add(key)
            out.append(query)
    return out[:limit]


def build_queries(p: Preprocessed) -> list[str]:
    return dedup_queries([p.standalone_question, *p.extra_queries])
