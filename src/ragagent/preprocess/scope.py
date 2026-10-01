"""LLM 只给说法；真实 doc_id 必须来自 indexed 登记表（§8）。"""

import logging
import sqlite3

from rapidfuzz.fuzz import partial_ratio

from ragagent.ingest.registry_db import RegistryDB, normalize_title
from ragagent.preprocess.filters import effective_filters
from ragagent.retrieval.filters import FILE_TYPES, validate_filters

log = logging.getLogger(__name__)


def _candidates(registry, phrase, limit):
    if isinstance(registry, RegistryDB):
        return registry.title_candidates(phrase, limit=limit)
    # Deps.registry 也接受调用者按请求创建的 sqlite.Connection。
    conn = registry
    rows = conn.execute("""SELECT d.doc_id,d.title,t.filename FROM documents d
        JOIN doc_titles t ON d.doc_id=t.doc_id WHERE d.status='indexed'""").fetchall()
    rows = [dict(zip(("doc_id", "title", "filename"), r)) for r in rows]
    exact = [r for r in rows if phrase in (normalize_title(r["title"]), normalize_title(r["filename"]))]
    if exact:
        return exact[:limit]
    if len(phrase) < 3:
        return []
    quoted = '"' + phrase.replace('"', '""') + '"'
    rows = conn.execute("""SELECT d.doc_id,d.title,t.filename FROM doc_titles t
        JOIN documents d ON d.doc_id=t.doc_id WHERE doc_titles MATCH ?
        AND d.status='indexed' ORDER BY t.rank LIMIT ?""", (quoted, limit)).fetchall()
    return [dict(zip(("doc_id", "title", "filename"), r)) for r in rows]


def match_titles(titles, registry, cfg) -> tuple[list[str], list[str]]:
    ids, matched = set(), []
    for title in titles:
        phrase = normalize_title(title)
        if not phrase:
            continue
        candidates = _candidates(registry, phrase, max(cfg.fts_candidates, cfg.max_docs + 1))
        exact = [r for r in candidates if phrase in (normalize_title(r["title"]), normalize_title(r["filename"]))]
        valid = exact if exact else [r for r in candidates if max(
            partial_ratio(phrase, normalize_title(r["title"])),
            partial_ratio(phrase, normalize_title(r["filename"]))) >= cfg.fuzzy_threshold]
        hits = {r["doc_id"] for r in valid}
        if 0 < len(hits) <= cfg.max_docs:
            ids.update(hits)
            matched.append(title)
        else:
            log.info("scope title ignored matches=%d", len(hits))
    if len(ids) > cfg.max_docs:
        log.info("scope combined titles ignored matches=%d", len(ids))
        return [], []
    return sorted(ids), matched


def resolve_scope(scope, registry, cfg, hard=None):
    """返回 (soft_filters, scope_note, 生效scope_doc_ids)。硬过滤冲突则整套soft丢弃。"""
    file_types = set()
    for saying in scope.file_types:
        for synonym in cfg.file_type_synonyms:
            if normalize_title(saying) in {normalize_title(w) for w in synonym.words}:
                file_types.update(set(synonym.types) & FILE_TYPES)
    ids, titles = [], []
    if scope.doc_titles:
        if registry is None:
            log.warning("scope registry unavailable; skip soft filters")
            return None, None, []
        try:
            ids, titles = match_titles(scope.doc_titles, registry, cfg)
        except (sqlite3.Error, OSError):
            log.warning("scope registry unavailable; skip soft filters")
            return None, None, []
    soft = {}
    if ids:
        soft["doc_ids"] = ids
    if file_types:
        soft["file_types"] = sorted(file_types)
    if not soft:
        return None, None, []
    soft = validate_filters(soft)
    merged = effective_filters(hard, soft)
    # 任意键冲突都会让整套soft作废，而不是仅丢弃冲突键。
    if any(hard and hard.get(k) and v and not (set(hard[k]) & set(v)) for k, v in soft.items()):
        return None, None, []
    parts = [f"《{t}》" for t in titles]
    if file_types:
        parts.append("/".join(sorted(file_types)) + "文件")
    return soft, "已限定在" + "、".join(parts) + "中检索", (merged or {}).get("doc_ids", [])
