"""文档级幂等写入：先完成新版本全部upsert，成功后才删除旧版本。"""
from __future__ import annotations

from itertools import batched
from math import isfinite
import re

from ragagent.config import Index
from ragagent.index.models import EmbedItem, to_row
from ragagent.ingest.models import CleanDocument
from ragagent.ingest.chunking.models import Chunk, make_chunk_id

DOC_ID_RE = re.compile(r'^[0-9a-f]{16}$')
LIMITS = {'chunk_id': 64, 'doc_id': 32, 'type': 16, 'text': 65535,
          'embed_text': 65535, 'content_hash': 64, 'title': 512, 'section_path': 2048,
          'source_path': 1024, 'file_type': 16, 'image_path': 1024, 'prev_id': 64, 'next_id': 64}


def _doc_id(doc_id: str) -> str:
    if not isinstance(doc_id, str) or not DOC_ID_RE.fullmatch(doc_id):
        raise ValueError(f'非法doc_id: {doc_id!r}')
    return doc_id


def _validate_row(row: dict):
    for field, maximum in LIMITS.items():
        if len(row[field].encode('utf-8')) > maximum:
            raise ValueError(f'{field}超过Milvus字节上限{maximum}')
    for field in ('image_ids', 'related_ids'):
        if len(row[field]) > 32 or any(len(s.encode('utf-8')) > 64 for s in row[field]):
            raise ValueError(f'{field}超过Milvus ARRAY限制')


def index_document(client, coll: str, emb, doc: CleanDocument, chunks: list[Chunk],
                   *, cfg: Index | None = None) -> int:
    doc_id = _doc_id(doc.doc_id)
    version = doc.version
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError('version必须是正整数')
    if not chunks:  # 空/失败文档不得删掉先前可用版本
        return 0
    rows = []
    seen = set()
    for chunk in chunks:
        if chunk.doc_id != doc_id or chunk.version != version:
            raise ValueError('chunk不属于当前文档/版本')
        if chunk.seq < 0 or chunk.chunk_id != make_chunk_id(doc_id, version, chunk.seq):
            raise ValueError('chunk_id与文档版本/seq不匹配')
        if chunk.chunk_id in seen:
            raise ValueError('chunk_id重复')
        seen.add(chunk.chunk_id)
        if chunk.type == 'image':
            EmbedItem(image_path=chunk.image_path, text=chunk.embed_text)
        row = to_row(chunk, doc, [])
        _validate_row(row)
        rows.append(row)
    texts = [c for c in chunks if c.type != 'image']
    images = [c for c in chunks if c.type == 'image']
    vectors = {}
    for group, size, encode in ((texts, cfg.text_batch if cfg else 32, emb.text),
                               (images, cfg.image_batch if cfg else 8, emb.image)):
        for batch in batched(group, size):
            inputs = ([(c.image_path, c.embed_text) for c in batch] if group is images
                      else [c.embed_text for c in batch])
            result = encode(inputs)
            if len(result) != len(batch):
                raise ValueError('编码结果数量不匹配')
            for chunk, vector in zip(batch, result, strict=True):
                if (not vector or not all(isfinite(x) for x in vector)
                        or len(vector) != (cfg.dim if cfg else emb.dim)):
                    raise ValueError('向量维度错误或非有限')
                vectors[chunk.chunk_id] = vector
    for row in rows:
        row['dense'] = vectors[row['chunk_id']]
    for batch in batched(rows, cfg.insert_batch if cfg else 500):
        result = client.upsert(coll, list(batch))
        if result.get('upsert_count', len(batch)) != len(batch):
            raise RuntimeError('Milvus未确认全部upsert成功，不删除旧版本')
    client.delete(coll, filter=f'doc_id == "{doc_id}" and version < {version}')
    return len(rows)


def delete_document(client, coll: str, doc_id: str):
    return client.delete(coll, filter=f'doc_id == "{_doc_id(doc_id)}"')


def flush(client, coll: str):
    """批次结束调用一次，不对每份文档flush。"""
    return client.flush(coll)
