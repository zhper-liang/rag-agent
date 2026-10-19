import hashlib
from pathlib import Path

import pytest

from ragagent.ingest.pipeline import run_clean
from ragagent.ingest.registry_db import RegistryDB, make_doc_id
from ragagent.ingest.upload import UploadStore, sanitize_filename, sanitize_names
from tests.index.helpers import config
from tests.ingest.test_pipeline import article


def upload_config(tmp_path, **updates):
    paths = {f'paths.{k}': str(tmp_path / k) for k in
             ('raw', 'uploads', 'parsed', 'clean', 'chunks', 'registry_db', 'session_db', 'feedback_db', 'ingest_lock')}
    return config(**paths, **updates)


@pytest.mark.parametrize('name,expected', [
    ('../../文件.MD', '文件.md'), (r'C:\\tmp\\资料.PDF', '资料.pdf'),
    ('\x00\x1f 文\t件 :*?"<>|.TXT ', '文件.txt'), ('...hello...md', 'hello.md'),
    ('e\u0301.md', 'é.md'), ('a' * 30 + '.PDF', 'a' * 8 + '.pdf'),
    (' A   B.md ', 'A B.md'),
])
def test_names(tmp_path, name, expected):
    assert sanitize_filename(name, upload_config(tmp_path, **{'api.upload.max_name_chars': 8})) == expected


@pytest.mark.parametrize('name', ['', '..', '.md', 'x.exe', '/a/', ' *?.txt'])
def test_invalid_names(tmp_path, name):
    with pytest.raises(ValueError):
        sanitize_filename(name, upload_config(tmp_path))


def test_duplicate_names_after_normalizing(tmp_path):
    with pytest.raises(ValueError):
        sanitize_names(['a.MD', '../a.md'], upload_config(tmp_path))


def test_source_prefix_missing_and_selected_scope(tmp_path):
    cfg = upload_config(tmp_path)
    raw, uploads = Path(cfg.paths.raw), Path(cfg.paths.uploads)
    raw.mkdir(); uploads.mkdir()
    (raw / 'same.md').write_text(article('原始部署'))
    (uploads / 'same.md').write_text(article('上传查询'))
    run_clean(cfg=cfg)
    run_clean(uploads, cfg=cfg, source_prefix='@uploads')
    raw_id, up_id = make_doc_id('same.md'), make_doc_id('@uploads/same.md')
    assert raw_id != up_id
    (raw / 'same.md').unlink()
    run_clean(cfg=cfg)
    with RegistryDB(cfg.paths.registry_db) as db:
        assert db.get(raw_id)['status'] == 'deleted'
        assert db.get(up_id)['status'] == 'cleaned'
    (uploads / 'same.md').unlink()
    (uploads / 'only.md').write_text(article('隔离范围'))
    run_clean(uploads, cfg=cfg, source_prefix='@uploads', source_names={'only.md'})
    with RegistryDB(cfg.paths.registry_db) as db:
        assert db.get(up_id)['status'] == 'cleaned'  # 局部任务不能 mark_missing。
    run_clean(uploads, cfg=cfg, source_prefix='@uploads')
    with RegistryDB(cfg.paths.registry_db) as db:
        assert db.get(up_id)['status'] == 'deleted'
        assert db.get(make_doc_id('@uploads/only.md'))['status'] == 'cleaned'


def test_job_recovery_pruning_cascade_and_conflicts(tmp_path):
    cfg = upload_config(tmp_path, **{'api.upload.jobs_keep': 2})
    store = UploadStore(cfg)
    digest = hashlib.sha256(b'old').hexdigest()
    store.create('one', [('a.md', digest)])
    assert store.conflicts([('a.md', 'different')])[0]['name'] == 'a.md'
    assert not store.conflicts([('a.md', digest)])
    store.state('one', 'cleaning')
    store.recover()
    assert store.get('one')['status'] == 'interrupted'
    assert store.retry('one')
    assert not store.retry('one')
    store.state('one', 'done')
    folder = store.staging / 'one'; folder.mkdir(parents=True)
    (folder / '0').write_bytes(b'old')
    store.create('two', [('b.md', digest)])
    store.create('three', [('c.md', digest)])
    assert store.get('one') is None and not folder.exists()
    with RegistryDB(cfg.paths.registry_db) as db:
        assert db.conn.execute("SELECT count(*) FROM upload_files WHERE job_id='one'").fetchone()[0] == 0


def test_manifest_backfill_and_unchanged_schema(tmp_path):
    cfg = upload_config(tmp_path)
    with RegistryDB(cfg.paths.registry_db) as db:
        record, _ = db.register('old.md', 'a' * 64)
        db.set_status(record['doc_id'], 'indexed')
        original = [r['name'] for r in db.conn.execute('PRAGMA table_info(documents)')]
        with db.transaction():
            db.conn.execute('CREATE TABLE index_manifest(doc_id TEXT, version INTEGER, updated_at TEXT)')
            db.conn.execute('INSERT INTO index_manifest VALUES (?,?,?)', (record['doc_id'], 1, '2026-10-18T01:00:00+00:00'))
    with RegistryDB(cfg.paths.registry_db) as db:
        assert [r['name'] for r in db.conn.execute('PRAGMA table_info(documents)')] == original
        assert db.conn.execute('SELECT indexed_at FROM doc_times').fetchone()[0] == '2026-10-18T01:00:00+00:00'
        with db.transaction():
            db.conn.execute("UPDATE doc_times SET indexed_at='2026-10-17T00:00:00+00:00'")
    with RegistryDB(cfg.paths.registry_db) as db:
        assert db.conn.execute('SELECT indexed_at FROM doc_times').fetchone()[0].startswith('2026-10-17')
