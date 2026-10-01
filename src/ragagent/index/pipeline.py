"""登记表驱动的离线写入入口：run_index(chunk_dir)，支持argparse CLI。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ragagent.config import Config, load_config
from ragagent.index.embedder import Embedder
from ragagent.index.milvus import connect, ensure_collection, load, wait_indexed
from ragagent.index.writer import delete_document, flush, index_document
from ragagent.ingest.chunking.models import Chunk
from ragagent.ingest.models import CleanDocument
from ragagent.ingest.registry_db import RegistryDB


def run_index(chunk_dir: str | Path | None = None, *, cfg: Config | None = None,
              client=None, emb=None, registry: RegistryDB | None = None,
              collection: str | None = None, wait: bool = True, force: bool = False) -> dict:
    cfg = cfg or load_config()
    chunk_dir = Path(chunk_dir or cfg.paths.chunks)
    coll = collection or cfg.services.milvus.collection
    own_client, own_emb, own_registry = client is None, emb is None, registry is None
    client = client if client is not None else connect(cfg)
    emb = emb if emb is not None else Embedder(cfg=cfg)
    registry = registry if registry is not None else RegistryDB(cfg.paths.registry_db)
    report = {'collection': coll, 'documents': 0, 'chunks': 0, 'deleted': 0,
              'skipped': 0, 'errors': {}, 'indexes': {}}
    try:
        created = ensure_collection(client, coll, cfg.index.dim)
        # 私有扩展表，不改变P0 documents列或消费者读取契约。
        registry.conn.execute('''CREATE TABLE IF NOT EXISTS index_manifest (
            doc_id TEXT NOT NULL, collection_name TEXT NOT NULL, version INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL, embed_model TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY(doc_id, collection_name))''')
        registry.conn.commit()
        version = emb.version  # 模型服务不可达时不改任何文档状态
        for record in registry.list_documents():
            doc_id, v = record['doc_id'], record['version']
            if record['status'] == 'deleted':
                delete_document(client, coll, doc_id)
                with registry.conn:
                    registry.conn.execute('DELETE FROM index_manifest WHERE doc_id=? AND collection_name=?',
                                          (doc_id, coll))
                report['deleted'] += 1
                continue
            if record['status'] not in ('cleaned', 'indexed') and not (
                    record['status'] == 'failed' and (record['error'] or '').startswith('index:')):
                report['skipped'] += 1
                continue
            previous = registry.conn.execute('''SELECT version,embed_model FROM index_manifest
                WHERE doc_id=? AND collection_name=?''', (doc_id, coll)).fetchone()
            if not force and not created and previous and previous['version'] == v and previous['embed_model'] == version:
                report['skipped'] += 1
                continue
            try:
                clean_path = Path(cfg.paths.clean) / doc_id / f'v{v}' / 'clean.json'
                doc = CleanDocument.model_validate_json(clean_path.read_text(encoding='utf-8'))
                if (doc.doc_id, doc.version, doc.content_hash) != (doc_id, v, record['content_hash']):
                    raise ValueError('clean.json与登记表文档ID/版本/hash不一致')
                chunks = [Chunk.model_validate_json(line) for line in
                          (chunk_dir / doc_id / f'v{v}.jsonl').read_text(encoding='utf-8').splitlines()
                          if line.strip()]
                n = index_document(client, coll, emb, doc, chunks, cfg=cfg.index)
                if not n:
                    raise ValueError('清洗状态文档没有任何chunk，保留旧版本')
                # set_status自身的事务一并提交manifest与FTS更新。
                registry.conn.execute('''INSERT INTO index_manifest VALUES (?,?,?,?,?,?)
                    ON CONFLICT(doc_id,collection_name) DO UPDATE SET version=excluded.version,
                    chunk_count=excluded.chunk_count,embed_model=excluded.embed_model,
                    updated_at=excluded.updated_at''',
                    (doc_id, coll, v, n, version, datetime.now(timezone.utc).isoformat()))
                registry.set_status(doc_id, 'indexed', error=None)
                report['documents'] += 1
                report['chunks'] += n
            except Exception as exc:
                registry.conn.rollback()
                message = f'index:{type(exc).__name__}: {exc}'
                registry.set_status(doc_id, 'failed', error=message)
                report['errors'][doc_id] = message
        if report['documents'] or report['deleted']:
            flush(client, coll)
        if wait and report['chunks']:
            if created:
                # 只release本次新建的集合；从sealed索引重新load，排除growing暴力查询。
                client.release_collection(coll)
            report['indexes'] = wait_indexed(client, coll)
            load(client, coll)
        return report
    finally:
        if own_emb:
            emb.close()
        if own_client:
            client.close()
        if own_registry:
            registry.close()


def main():
    parser = argparse.ArgumentParser(description='登记表驱动的Chunk JSONL→Milvus写入')
    parser.add_argument('chunk_dir', type=Path, nargs='?')
    parser.add_argument('--collection', help='新建/重建时指定collection；L1仅用l1_前缀')
    parser.add_argument('--embed-url', help='覆盖本次HTTP编码服务地址（联调用桩）')
    parser.add_argument('--force', action='store_true', help='幂等重写已登记的文档')
    parser.add_argument('--no-wait', action='store_true', help='只flush，不等待索引或load')
    args = parser.parse_args()
    cfg = load_config()
    with Embedder(args.embed_url, cfg=cfg) as emb:
        report = run_index(args.chunk_dir, cfg=cfg, emb=emb, collection=args.collection,
                           wait=not args.no_wait, force=args.force)
    print(json.dumps(report, ensure_ascii=False, default=str))
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
