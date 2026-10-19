"""入库日期解析；内部文档范围不借用用户 filters 的百项白名单。"""
from datetime import date
import re

from fastapi import HTTPException

from ragagent.ingest.registry_db import RegistryDB
from ragagent.retrieval.filters import DOC_ID_RE


def resolve_time_docs(cfg, time_from, time_to, filters):
    if time_from is None and time_to is None:
        return None
    bounds = []
    for value in (time_from, time_to):
        if value is None:
            bounds.append(None)
            continue
        try:
            if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
                raise ValueError
            bounds.append(date.fromisoformat(value))
        except ValueError:
            raise HTTPException(400, '时间必须为有效的 YYYY-MM-DD 日期') from None
    start, end = bounds
    if start is not None and end is not None and start >= end:
        raise HTTPException(400, 'time_from 必须早于 time_to')
    parts, args = ["d.status='indexed'"], []
    for op, value in (('>=', time_from), ('<', time_to)):
        if value is not None:
            parts.append(f't.indexed_at {op} ?')
            args.append(value)
    ids_filter = (filters or {}).get('doc_ids')
    if ids_filter:
        parts.append('d.doc_id IN (' + ','.join('?' for _ in ids_filter) + ')')
        args.extend(ids_filter)
    with RegistryDB(cfg.paths.registry_db, readonly=True) as db:
        rows = db.conn.execute('SELECT d.doc_id FROM documents d JOIN doc_times t USING(doc_id) WHERE '
            + ' AND '.join(parts) + ' ORDER BY d.doc_id LIMIT ?',
            (*args, cfg.api.time_filter_max_docs + 1)).fetchall()
    if not rows:
        raise HTTPException(400, '该时间范围内没有文档')
    if len(rows) > cfg.api.time_filter_max_docs:
        raise HTTPException(400, '时间范围内文档过多，请缩小范围')
    ids = [r[0] for r in rows]
    if not all(DOC_ID_RE.fullmatch(i) for i in ids):
        raise HTTPException(400, '登记表文档标识无效')
    return ids
