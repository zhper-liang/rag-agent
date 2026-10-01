from pathlib import Path

import pytest

from ragagent.index.pipeline import run_index
from ragagent.ingest.registry_db import RegistryDB
from tests.index.helpers import Client, Encoder, chunks, config, document


def write_artifacts(cfg, doc):
    folder = Path(cfg.paths.clean) / doc.doc_id / f'v{doc.version}'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'clean.json').write_text(doc.model_dump_json(), encoding='utf-8')
    path = Path(cfg.paths.chunks) / doc.doc_id / f'v{doc.version}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = chunks(doc)
    path.write_text(''.join(c.model_dump_json() + '\n' for c in parts), encoding='utf-8')
    return parts


def prepare(tmp_path):
    cfg = config(**{'paths.clean': str(tmp_path / 'clean'), 'paths.chunks': str(tmp_path / 'chunks'),
                    'paths.registry_db': str(tmp_path / 'registry.db'), 'index.dim': 4})
    registry = RegistryDB(cfg.paths.registry_db)
    record, _ = registry.register('manual.md', 'a' * 64)
    doc = document().model_copy(update={'doc_id': record['doc_id']})
    parts = write_artifacts(cfg, doc)
    registry.set_status(doc.doc_id, 'cleaned')
    return cfg, registry, doc, parts


def test_pipeline_manifest_status_skip_and_delete(tmp_path, monkeypatch):
    monkeypatch.setattr('ragagent.index.pipeline.ensure_collection', lambda *_: False)
    cfg, registry, doc, parts = prepare(tmp_path)
    client, emb = Client(), Encoder()
    with registry:
        result = run_index(cfg=cfg, registry=registry, client=client, emb=emb,
                           collection='l1_test', wait=False)
        assert not result['errors'] and result['documents'] == 1
        assert result['chunks'] == len(parts)
        assert registry.get(doc.doc_id)['status'] == 'indexed'
        stats = registry.conn.execute('SELECT * FROM index_manifest').fetchone()
        assert stats['chunk_count'] == len(parts) and stats['embed_model'] == 'test@4'
        assert len(registry.title_candidates('manual')) == 1
        assert len([c for c in client.calls if c[0] == 'flush']) == 1
        result = run_index(cfg=cfg, registry=registry, client=client, emb=emb,
                           collection='l1_test', wait=False)
        assert result['documents'] == 0 and result['skipped'] == 1
        # 新collection绝不能因为登记表全局indexed就跳过。
        result = run_index(cfg=cfg, registry=registry, client=client, emb=emb,
                           collection='l1_new', wait=False)
        assert result['documents'] == 1
        registry.delete(doc.doc_id)
        result = run_index(cfg=cfg, registry=registry, client=client, emb=emb,
                           collection='l1_test', wait=False)
        assert result['deleted'] == 1
        assert client.calls[-2][2] == f'doc_id == "{doc.doc_id}"'
        assert registry.conn.execute('SELECT count(*) FROM index_manifest WHERE collection_name="l1_test"').fetchone()[0] == 0


def test_pipeline_failure_not_indexed_and_retries(tmp_path, monkeypatch):
    monkeypatch.setattr('ragagent.index.pipeline.ensure_collection', lambda *_: False)
    cfg, registry, doc, parts = prepare(tmp_path)
    client, emb = Client(fail_batch=1), Encoder()
    with registry:
        result = run_index(cfg=cfg, registry=registry, client=client, emb=emb,
                           collection='l1_test', wait=False)
        assert doc.doc_id in result['errors']
        assert registry.get(doc.doc_id)['status'] == 'failed'
        assert registry.conn.execute('SELECT count(*) FROM index_manifest').fetchone()[0] == 0
        assert not any(c[0] in ('delete', 'flush') for c in client.calls)
        client.fail_batch = None
        result = run_index(cfg=cfg, registry=registry, client=client, emb=emb,
                           collection='l1_test', wait=False)
        assert result['documents'] == 1 and registry.get(doc.doc_id)['status'] == 'indexed'


def test_pipeline_rejects_artifact_version_or_hash(tmp_path, monkeypatch):
    monkeypatch.setattr('ragagent.index.pipeline.ensure_collection', lambda *_: False)
    cfg, registry, doc, _ = prepare(tmp_path)
    write_artifacts(cfg, doc.model_copy(update={'content_hash': 'b' * 64}))
    client, emb = Client(), Encoder()
    with registry:
        result = run_index(cfg=cfg, registry=registry, client=client, emb=emb,
                           collection='l1_test', wait=False)
        assert '不一致' in result['errors'][doc.doc_id]
        assert client.calls == [] and emb.calls == []


def test_recreated_collection_does_not_trust_old_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr('ragagent.index.pipeline.ensure_collection', lambda *_: False)
    cfg, registry, doc, parts = prepare(tmp_path)
    with registry:
        client, emb = Client(), Encoder()
        first = run_index(cfg=cfg, registry=registry, client=client, emb=emb,
                          collection='l1_test', wait=False)
        assert first['documents'] == 1
        monkeypatch.setattr('ragagent.index.pipeline.ensure_collection', lambda *_: True)
        result = run_index(cfg=cfg, registry=registry, client=client, emb=emb,
                           collection='l1_test', wait=False)
        assert result['documents'] == 1 and result['chunks'] == len(parts)
