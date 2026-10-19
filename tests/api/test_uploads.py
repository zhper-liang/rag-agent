import asyncio
from dataclasses import replace
import hashlib
from pathlib import Path
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from langgraph.runtime import Runtime

from ragagent.api.app import create_app
from ragagent.api.time_filters import resolve_time_docs
from ragagent.graph.deps import Deps
from ragagent.graph.nodes.retrieve import retrieve
from ragagent.ingest.lock import ingest_lock
from ragagent.ingest.registry_db import RegistryDB, make_doc_id
from ragagent.ingest.upload import UploadManager
from ragagent.retrieval.filters import build_expr, validate_filters
from tests.ingest.test_upload import upload_config
from tests.ingest.test_pipeline import article

HEADERS = {'X-RAG-Client': 'web'}


def client_for(cfg):
    deps = Deps(cfg=cfg, count=len)
    return TestClient(create_app(cfg, deps=deps, graph=object(), trace_client=False), base_url='http://127.0.0.1')


def wait_job(client, job_id):
    for _ in range(300):
        response = client.get('/api/uploads/' + job_id, headers=HEADERS)
        assert response.status_code == 200
        job = response.json()
        if job['status'] in ('done', 'failed', 'interrupted'):
            return job
        time.sleep(.01)
    raise AssertionError('任务没有结束')


@pytest.fixture
def cfg(tmp_path):
    c = upload_config(tmp_path, **{'api.upload.lock_retry_s': .02})
    return c.model_copy(update={'monitor': c.monitor.model_copy(update={
        'langfuse': c.monitor.langfuse.model_copy(update={'enabled': False})})})


@pytest.mark.parametrize('headers', [{}, {'X-RAG-Client': 'other'}, {**HEADERS, 'Host': 'evil.test'},
    {**HEADERS, 'Origin': 'http://evil.test'}, {**HEADERS, 'Host': '127.0.0.1.evil.test'},
    {**HEADERS, 'Host': 'localhost@evil.test'}, {**HEADERS, 'Host': '['},
    {**HEADERS, 'Origin': 'null'}, {**HEADERS, 'Origin': 'http://['}])
def test_access_before_body(cfg, headers):
    with client_for(cfg) as client:
        for method, url in [('post', '/api/uploads'), ('get', '/api/uploads'), ('post', '/api/uploads/fake/retry')]:
            assert getattr(client, method)(url, headers=headers).status_code == 403
    assert not Path(cfg.paths.uploads).exists()


def test_conflict_rejects_entire_request_without_files_or_jobs(cfg):
    with RegistryDB(cfg.paths.registry_db) as db:
        record, _ = db.register('@uploads/a.md', hashlib.sha256(b'old').hexdigest(), title='旧标题')
        db.set_status(record['doc_id'], 'indexed')
    with client_for(cfg) as client:
        response = client.post('/api/uploads', files=[('files', ('a.md', b'new')), ('files', ('other.md', b'x'))], headers=HEADERS)
        assert response.status_code == 409
        assert response.json()['conflicts'] == [dict(name='a.md', doc_id=record['doc_id'], title='旧标题', version=1)]
        assert client.get('/api/uploads', headers=HEADERS).json() == {'jobs': []}
    assert not Path(cfg.paths.uploads).exists()
    assert not (Path(cfg.paths.parsed) / '.uploads').exists()


@pytest.mark.parametrize('files,data', [([('files', ('a.md', b'x')), ('files', ('../a.MD', b'y'))], {}),
    ([('files', ('bad.exe', b'x'))], {}), ([('files', ('a.md', b'x'))], {'overwrite': 'yes'}),
    ([('unknown', ('a.md', b'x'))], {}), ([], {'overwrite': 'false'})])
def test_invalid_upload(cfg, files, data):
    with client_for(cfg) as client:
        response = client.post('/api/uploads', files=files, data=data, headers=HEADERS)
        assert response.status_code == 400
        assert client.get('/api/uploads', headers=HEADERS).json()['jobs'] == []


def test_limit_during_multipart_stream(cfg):
    cfg = cfg.model_copy(update={'api': cfg.api.model_copy(update={'upload': cfg.api.upload.model_copy(update={'max_file_mb': 1, 'max_files': 1})})})
    with client_for(cfg) as client:
        assert client.post('/api/uploads', files={'files': ('big.md', b'a' * (1024 * 1024 + 1))}, headers=HEADERS).status_code == 400
        assert client.post('/api/uploads', files=[('files', ('a.md', b'x')), ('files', ('b.md', b'y'))], headers=HEADERS).status_code == 400
    assert not Path(cfg.paths.uploads).exists()


def fake_pipeline(monkeypatch):
    from ragagent.ingest.chunking.pipeline import run_chunk
    monkeypatch.setattr('ragagent.ingest.chunking.pipeline.run_chunk', lambda doc, **kw: run_chunk(doc, cfg=kw['cfg'], count=len, encode=lambda texts: [[1., 0.] for _ in texts]))
    def index(*args, cfg, registry, doc_ids, **kw):
        assert "client" in kw  # 入库必须复用在线 Milvus，不创建/关闭同 URI 连接。
        for doc_id in doc_ids:
            if registry.get(doc_id)['status'] == 'cleaned':
                registry.set_status(doc_id, 'indexed')
                with registry.transaction():
                    registry.conn.execute('INSERT OR REPLACE INTO doc_times VALUES (?,?)', (doc_id, '2026-10-20T00:00:00+00:00'))
    monkeypatch.setattr('ragagent.index.pipeline.run_index', index)


def test_upload_lock_queue_created_unchanged_overwrite_duplicate_and_docs(cfg, monkeypatch):
    fake_pipeline(monkeypatch)
    text = article('上传测试')
    with client_for(cfg) as client:
        with ingest_lock(cfg.paths.ingest_lock):
            response = client.post('/api/uploads', files={'files': ('a.md', text.encode())}, headers={**HEADERS, 'Origin': 'http://127.0.0.1'})
            assert response.status_code == 202
            job_id = response.json()['job_id']
            time.sleep(.06)
            assert client.get('/api/uploads/' + job_id, headers=HEADERS).json()['status'] == 'queued'
            assert not Path(cfg.paths.uploads).exists()
        assert wait_job(client, job_id)['files'][0]['result'] == 'created'
        doc_id = response.json()['files'][0]['doc_id']
        assert (Path(cfg.paths.chunks) / doc_id / 'v1.jsonl').is_file()
        assert client.get('/api/docs').json()['docs'][0]['indexed_at'] is not None
        response = client.post('/api/uploads', files={'files': ('a.md', text.encode())}, headers=HEADERS)
        assert wait_job(client, response.json()['job_id'])['files'][0]['result'] == 'unchanged'
        response = client.post('/api/uploads', files={'files': ('b.md', text.encode())}, headers=HEADERS)
        assert wait_job(client, response.json()['job_id'])['files'][0]['result'] == 'duplicate'
        assert client.post('/api/uploads', files={'files': ('a.md', article('全新主题').encode())}, headers=HEADERS).status_code == 409
        response = client.post('/api/uploads', files={'files': ('a.md', article('全新主题').encode())}, data={'overwrite': 'true'}, headers=HEADERS)
        assert wait_job(client, response.json()['job_id'])['files'][0]['result'] == 'updated'
        with RegistryDB(cfg.paths.registry_db) as db:
            assert db.get(doc_id)['version'] == 2
        assert client.post('/api/uploads/' + job_id + '/retry', headers=HEADERS).status_code == 409
        assert client.get('/api/uploads/not-found', headers=HEADERS).status_code == 404


def test_restart_interrupted_retry(cfg, monkeypatch):
    fake_pipeline(monkeypatch)
    store = UploadManager(cfg).store
    text = article('重启恢复').encode()
    store.create('restart', [('resume.md', hashlib.sha256(text).hexdigest())])
    folder = store.staging / 'restart'; folder.mkdir(parents=True)
    (folder / '0').write_bytes(text)
    store.state('restart', 'cleaning')
    with client_for(cfg) as client:
        assert client.get('/api/uploads/restart', headers=HEADERS).json()['status'] == 'interrupted'
        assert client.post('/api/uploads/restart/retry', headers=HEADERS).status_code == 202
        assert wait_job(client, 'restart')['status'] == 'done'


def seed_times(cfg, n=3):
    with RegistryDB(cfg.paths.registry_db) as db:
        ids = []
        for i in range(n):
            row, _ = db.register(f'{i}.md', f'{i:064x}')
            db.set_status(row['doc_id'], 'indexed')
            db.conn.execute('INSERT INTO doc_times VALUES (?,?)', (row['doc_id'], f'2026-10-{18 + i % 3:02d}T00:00:00+00:00'))
            ids.append(row['doc_id'])
        db.conn.commit()
    return ids


def test_time_bounds_intersection_and_more_than_100(cfg):
    ids = seed_times(cfg, 120)
    resolved = resolve_time_docs(cfg, '2026-10-18', '2026-10-21', None)
    assert len(resolved) == 120
    expr = build_expr(None, time_doc_ids=resolved)
    assert all(i in expr for i in ids)
    with pytest.raises(ValueError):
        validate_filters({'doc_ids': resolved})
    assert resolve_time_docs(cfg, None, '2026-10-19', {'doc_ids': ids[:3]}) == [ids[0]]
    for invalid in ([], ['invalid'], ['a' * 16 + '"] or true'], 'a' * 16):
        with pytest.raises(ValueError):
            build_expr(None, time_doc_ids=invalid)
    small = cfg.model_copy(update={'api': cfg.api.model_copy(update={'time_filter_max_docs': 2})})
    with pytest.raises(Exception) as err:
        resolve_time_docs(small, '2026-10-18', None, None)
    assert err.value.status_code == 400


@pytest.mark.parametrize('bounds', [{'time_from': '2026-02-30'}, {'time_from': '2026-1-1'},
    {'time_from': '2026-10-20', 'time_to': '2026-10-20'}, {'time_to': '2020-01-01'},
    {'time_from': '2026-10-21', 'time_to': '2026-10-18'}])
def test_time_errors_before_sse(cfg, bounds):
    seed_times(cfg)
    with client_for(cfg) as client:
        response = client.post('/api/chat', json={'question': '问题', 'session_id': 'time', **bounds})
        assert response.status_code == 400
        assert 'text/event-stream' not in response.headers.get('content-type', '')


def test_time_scope_survives_soft_conflicts_and_rewrite(cfg, monkeypatch):
    captured = []
    from ragagent.retrieval.models import RetrievalResult
    async def run(queries, filters, deps):
        captured.append(build_expr(filters, time_doc_ids=deps.time_doc_ids))
        return [], RetrievalResult()
    monkeypatch.setattr('ragagent.graph.nodes.retrieve.run_retrieve', run)
    runtime = Runtime(context=Deps(cfg=cfg, count=len, time_doc_ids=['a' * 16]), stream_writer=lambda _: None)
    state = {'queries': ['问题'], 'filters': {'file_types': ['md']}, 'soft_filters': {'doc_ids': ['b' * 16]}}
    asyncio.run(retrieve(state, runtime))
    state['use_soft'] = False
    asyncio.run(retrieve(state, runtime))
    assert all('"a' in e and '"b' not in e and 'file_type in ["md"]' in e for e in captured)


def test_retry_preserves_mineru_batch_and_hides_raw_errors(cfg, monkeypatch):
    fake_pipeline(monkeypatch)
    from ragagent.ingest.pipeline import run_clean as clean
    attempts = []
    class Cloud:
        async def parse_batch(self, jobs, **kwargs):
            attempts.append(dict(self.resume_batches))
            job = jobs[0]
            if len(attempts) == 1:
                self.on_batch('known-cloud-batch', jobs)
                return {job.doc_id: TimeoutError('secret token must not leak')}
            folder = job.out_dir
            folder.mkdir(parents=True, exist_ok=True)
            import json
            (folder / 'retry_content_list.json').write_text(json.dumps([
                {'type': 'text', 'text': article('云端恢复'), 'text_level': 0, 'page_idx': 0}]))
            return {job.doc_id: folder}
    cloud = Cloud()
    monkeypatch.setattr('ragagent.ingest.pipeline.run_clean', lambda *a, **kw: clean(*a, **kw, mineru=cloud))
    with client_for(cfg) as client:
        response = client.post('/api/uploads', files={'files': ('resume.pdf', b'synthetic-pdf')}, headers=HEADERS)
        job_id = response.json()['job_id']
        failed = wait_job(client, job_id)
        assert failed['status'] == 'failed'
        assert 'secret' not in str(failed)
        with RegistryDB(cfg.paths.registry_db) as db:
            row = db.get(response.json()['files'][0]['doc_id'])
            assert row['status'] == 'parsing' and row['mineru_batch_id'] == 'known-cloud-batch'
        assert client.post(f'/api/uploads/{job_id}/retry', headers=HEADERS).status_code == 202
        assert wait_job(client, job_id)['status'] == 'done'
        assert attempts[1] == {row['doc_id']: 'known-cloud-batch'}


def test_chat_time_scope_in_context_resets_next_request(cfg):
    ids = seed_times(cfg)
    calls = []
    class Graph:
        async def astream(self, state, *, context, **kw):
            calls.append((state, context.time_doc_ids))
            yield 'custom', {'type': 'token', 'text': '范围正确'}
    dep = Deps(cfg=cfg, count=len)
    app = create_app(cfg, deps=dep, graph=Graph())
    with TestClient(app, base_url='http://127.0.0.1') as client:
        response = client.post('/api/chat', json={'question': '问题', 'session_id': 'scoped',
            'time_from': '2026-10-18', 'time_to': '2026-10-19'})
        assert response.status_code == 200
        assert calls[-1][1] == [ids[0]]
        assert 'time_doc_ids' not in calls[-1][0]
        response = client.post('/api/chat', json={'question': '问题', 'session_id': 'scoped'})
        assert response.status_code == 200 and calls[-1][1] is None


def test_retry_duplicate_retries_index_cleanup(cfg, monkeypatch):
    fake_pipeline(monkeypatch)
    from ragagent.index.pipeline import run_index as fake_index
    failures = []
    def fail_once(*args, **kwargs):
        failures.append(1)
        if len(failures) == 1:
            raise RuntimeError('private indexing error')
        return fake_index(*args, **kwargs)
    text = article('重复文档重试').encode()
    with RegistryDB(cfg.paths.registry_db) as db:
        original, _ = db.register('original.md', hashlib.sha256(text).hexdigest())
        db.set_status(original['doc_id'], 'indexed')
    monkeypatch.setattr('ragagent.index.pipeline.run_index', fail_once)
    with client_for(cfg) as client:
        response = client.post('/api/uploads', files={'files': ('dup.md', text)}, headers=HEADERS)
        job_id = response.json()['job_id']
        assert wait_job(client, job_id)['status'] == 'failed'
        assert client.post(f'/api/uploads/{job_id}/retry', headers=HEADERS).status_code == 202
        assert wait_job(client, job_id)['status'] == 'done'
        assert len(failures) == 2
