import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ragagent.api.app import SessionLocks, create_app, initial_state
from ragagent.api.images import image_url, public_event
from ragagent.api.sse import encode_event, graph_events
from ragagent.graph.build import build_graph
from ragagent.ingest.registry_db import RegistryDB
from tests.integration.helpers import KEYS, api_client, config, deps, events, preprocess_script, rows, script


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in KEYS:
        monkeypatch.setenv(name, 'test')


def test_chat_skips_retrieval_and_session_feedback_isolated(tmp_path, llm_url, embed_url):
    cfg = config(tmp_path, llm_url)
    dep = deps(cfg, embed_url)
    preprocess_script(llm_url, intent='chat', question='你好')
    with api_client(cfg, dep) as client:
        response = client.post('/api/chat', json={'question': '你好'}, headers={'X-Session-ID': 'chat'})
        wire = events(response)
        assert response.headers['X-Session-ID'] == 'chat'
        assert wire[-1]['citations'] == wire[-1]['sources'] == []
        assert not dep.milvus.calls
        mid = wire[-1]['message_id']
        assert client.post('/api/feedback', headers={'X-Session-ID': 'other'}, json={'message_id': mid, 'rating': 1}).status_code == 404
        assert client.post('/api/feedback', headers={'X-Session-ID': 'chat'}, json={'message_id': mid, 'rating': 1}).status_code == 200


def test_clarify_saved_and_next_clarify_rerouted_to_rag(tmp_path, llm_url, embed_url):
    cfg = config(tmp_path, llm_url)
    dep = deps(cfg, embed_url)
    preprocess_script(llm_url, intent='clarify')
    with api_client(cfg, dep) as client:
        first = events(client.post('/api/chat', json={'question': '它怎么样？', 'session_id': 's'}))
        assert [e['type'] for e in first] == ['progress', 'token', 'final']
        assert first[1]['text'] == '您想了解哪一部分？'
        preprocess_script(llm_url, intent='clarify')
        second = events(client.post('/api/chat', json={'question': rows()[2]['text'], 'session_id': 's'}))
        assert 'meta' in [e['type'] for e in second]
        saved = client.portal.call(client.app.state.graph.aget_state, {'configurable': {'thread_id': 's'}})
        assert saved.values['intent'] == 'rag' and saved.values['prev_intent'] == 'clarify'
        assert len(saved.values['messages']) == 4


@pytest.mark.parametrize('case', ['retry', 'exhausted', 'empty_rewrite', 'rewrite_failure', 'degraded'])
def test_m3_real_l2_nodes_conditional_paths(tmp_path, llm_url, embed_url, case):
    cfg = config(tmp_path, llm_url)
    rerank_calls = []
    async def transport(req):
        data = json.loads(req.content)
        rerank_calls.append(data)
        if case == 'degraded':
            return httpx.Response(503, json={'detail': 'unavailable'})
        n = len(data.get('passages', data.get('pairs', [])))
        score = 1. if case == 'retry' and len(rerank_calls) > 1 else .1
        return httpx.Response(200, json={'scores': [score] * n})
    reranker = httpx.AsyncClient(base_url=embed_url, transport=httpx.MockTransport(transport))
    dep = deps(cfg, embed_url, rerank_client=reranker)
    steps = [{'tool_args': {'intent': 'rag', 'standalone_question': '测试低分查询'}}]
    if case == 'empty_rewrite':
        steps.append({'tool_args': {'queries': []}})
    elif case == 'rewrite_failure':
        steps.extend([{'status': 400}, {'status': 400}])
    elif case in ('retry', 'exhausted'):
        steps += [{'tool_args': {'queries': [f'新查询{i}']}} for i in range(2)]
    script(llm_url, '*', steps)
    with api_client(cfg, dep) as client:
        wire = events(client.post('/api/chat', json={'question': '测试低分查询', 'session_id': case}))
        assert wire[-1]['type'] == 'final' and not any(e['type'] == 'error' for e in wire)
        saved = client.portal.call(client.app.state.graph.aget_state, {'configurable': {'thread_id': case}})
        state = saved.values
        if case == 'retry':
            assert state['retry_count'] == 1 and state['gen_meta']['scene'] == 'rag'
        elif case == 'degraded':
            assert state['retrieval']['degraded'] and state['gen_meta']['scene'] == 'rag'
        else:
            assert state['gen_meta']['scene'] == 'fallback'
            assert state['retry_count'] == (2 if case == 'exhausted' else 1)
            assert '资料库中没有找到' in state['answer']
        assert len(state['messages']) == 2


def test_generation_both_endpoints_failed_error_no_final_or_history(tmp_path, llm_url, embed_url):
    cfg = config(tmp_path, llm_url)
    dep = deps(cfg, embed_url)
    script(llm_url, 'small', [{'tool_args': {'intent': 'chat', 'standalone_question': '你好'}}, {'status': 400}])
    script(llm_url, 'small_backup', [{'status': 401}])
    with api_client(cfg, dep) as client:
        wire = events(client.post('/api/chat', json={'question': '你好', 'session_id': 'failed'}))
        assert wire[-1] == {'type': 'error', 'message': '回答服务暂时不可用，请稍后再试'}
        assert not any(e['type'] in ('token', 'final') for e in wire)
        saved = client.portal.call(client.app.state.graph.aget_state, {'configurable': {'thread_id': 'failed'}})
        assert not saved.values.get('messages')
        assert saved.values['gen_meta']['status'] == 'failed'


@pytest.mark.parametrize('body,headers,code', [
    ({'question': 'q'}, {}, 400),
    ({'question': 'q', 'session_id': ''}, {'X-Session-ID': 's'}, 400),
    ({'question': 'q', 'session_id': 's'}, {'X-Session-ID': ''}, 400),
    ({'question': 'q', 'session_id': '../bad'}, {}, 400),
    ({'question': 'q', 'session_id': 'one'}, {'X-Session-ID': 'two'}, 400),
    ({'question': ' ', 'session_id': 's'}, {}, 422),
    ({'question': 'x' * 2001, 'session_id': 's'}, {}, 422),
    ({'question': 'q', 'session_id': 's', 'filters': {'expr': 'DROP'}}, {}, 422),
    ({'question': 'q', 'session_id': 's', 'filters': {'doc_ids': ['bad']}}, {}, 422),
    ({'question': 'q', 'session_id': 's', 'use_scope': 'false'}, {}, 422),
    ({'question': 'q', 'session_id': 's', 'use_scope': 0}, {}, 422),
])
def test_request_validation_never_starts_graph(tmp_path, embed_url, body, headers, code):
    cfg = config(tmp_path)
    dep = deps(cfg, embed_url)
    with api_client(cfg, dep) as client:
        assert client.post('/api/chat', json=body, headers=headers).status_code == code
        assert not dep.milvus.calls


def populate_registry(cfg, doc_id='0123456789abcdef'):
    with RegistryDB(cfg.paths.registry_db) as db:
        # 与真实RegistryDB契约相同；API只读取登记版本，不用外部任意source_path。
        db.conn.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (doc_id, '../../secret.txt', '手册', 'md', 'text', 'hash', 1, 'indexed', None, 1, None, None, '2026-01-01'))
        db.conn.commit()
    path = Path(cfg.paths.clean) / doc_id / 'v1'
    path.mkdir(parents=True, exist_ok=True)
    (path / 'full.md').write_text('公开清洗原文')
    return path


def test_doc_download_registry_version_invalid_id_and_symlink(tmp_path, embed_url):
    cfg = config(tmp_path)
    dep = deps(cfg, embed_url)
    folder = populate_registry(cfg)
    with api_client(cfg, dep) as client:
        good = client.get('/api/docs/0123456789abcdef/file')
        assert good.status_code == 200 and good.text == '公开清洗原文'
        for invalid in ['not-an-id', '0123456789ABCDEF', '0123456789abcdef%0A', '0123456789abcdef0', '0123456789abcdef%27']:
            assert client.get(f'/api/docs/{invalid}/file').status_code == 400
        assert client.get('/api/docs/ffffffffffffffff/file').status_code == 404
        (folder / 'full.md').unlink()
        (folder / 'full.md').symlink_to(tmp_path / 'secret.txt')
        (tmp_path / 'secret.txt').write_text('secret')
        assert client.get('/api/docs/0123456789abcdef/file').status_code == 404


def test_doc_inline_view_is_sandboxed_plain_text(tmp_path, embed_url):
    cfg = config(tmp_path)
    populate_registry(cfg)
    with api_client(cfg, deps(cfg, embed_url)) as client:
        view = client.get('/api/docs/0123456789abcdef/file?inline=true')
        assert view.status_code == 200 and view.text == '公开清洗原文'
        assert view.headers['content-type'].startswith('text/plain') and view.headers['content-disposition'].startswith('inline')
        assert 'sandbox' in view.headers['content-security-policy'] and view.headers['x-content-type-options'] == 'nosniff'
        assert client.get('/api/docs/0123456789abcdef/file').headers['content-disposition'].startswith('attachment')


def test_doc_list_indexed_only_no_paths_search_and_truncation(tmp_path, embed_url):
    cfg = config(tmp_path)
    with api_client(cfg, deps(cfg, embed_url)) as client:
        assert client.get('/api/docs').json() == {'docs': [], 'truncated': False}
    populate_registry(cfg)
    with RegistryDB(cfg.paths.registry_db) as db:
        for i, status in enumerate(['indexed', 'indexed', 'cleaned', 'deleted']):
            doc, _ = db.register(f'docs/部署指南{i}.pdf', f'h{i}', file_type='pdf', parser='mineru-vlm', title=f'部署指南{i}')
            db.update(doc['doc_id'], status=status, pages=3)
    cfg = cfg.model_copy(update={'api': cfg.api.model_copy(update={'docs_list_limit': 2})})
    with api_client(cfg, deps(cfg, embed_url)) as client:
        full = client.get('/api/docs').json()
        assert full['truncated'] and len(full['docs']) == 2
        assert set(full['docs'][0]) == {'doc_id', 'title', 'file_type', 'pages', 'version'}
        assert 'secret' not in json.dumps(full) and 'source_path' not in json.dumps(full)
        hits = client.get('/api/docs', params={'q': '部署指南1'}).json()
        assert [d['title'] for d in hits['docs']] == ['部署指南1'] and not hits['truncated']
        assert client.get('/api/docs', params={'q': '部署指南2'}).json()['docs'] == []
        assert client.get('/api/docs', params={'q': 'x' * 2001}).status_code == 422


def test_use_scope_reaches_graph_state(tmp_path, llm_url, embed_url):
    cfg = config(tmp_path, llm_url)
    preprocess_script(llm_url, intent='chat', question='你好')
    with api_client(cfg, deps(cfg, embed_url)) as client:
        events(client.post('/api/chat', json={'question': '你好', 'session_id': 'u', 'use_scope': False}))
        saved = client.portal.call(client.app.state.graph.aget_state, {'configurable': {'thread_id': 'u'}})
        assert saved.values['use_scope'] is False and saved.values['soft_filters'] is None
    assert initial_state('q', None)['use_scope'] is True


def test_images_url_mapping_download_path_guards(tmp_path, embed_url):
    cfg = config(tmp_path)
    dep = deps(cfg, embed_url)
    folder = populate_registry(cfg)
    (folder / 'images').mkdir()
    (folder / 'images' / '图1.jpg').write_bytes(b'fake-jpeg')
    image = {'chunk_id': 'c', 'image_path': 'clean/0123456789abcdef/v1/images/图1.jpg', 'caption': '说明'}
    out = public_event({'type': 'meta', 'attachments': [image]})['attachments'][0]
    assert 'image_path' not in out and out['url'].endswith('%E5%9B%BE1.jpg')
    assert image_url('clean/0123456789abcdef/v1/images/../../secret.jpg') is None
    assert image_url('clean/0123456789abcdef/v1/images/script.svg') is None
    with api_client(cfg, dep) as client:
        assert client.get(out['url']).content == b'fake-jpeg'
        assert client.get('/api/images/0123456789abcdef/v2/图1.jpg').status_code == 404
        assert client.get('/api/images/0123456789ABCDEF/v1/图1.jpg').status_code == 400
        (tmp_path / 'outside.jpg').write_bytes(b'secret')
        (folder / 'images' / 'link.jpg').symlink_to(tmp_path / 'outside.jpg')
        assert client.get('/api/images/0123456789abcdef/v1/link.jpg').status_code == 404


def test_sse_newlines_are_json_encoded():
    data = encode_event({'type': 'token', 'text': 'a\n\ndata: {bad}😀'})
    assert data.count(b'\n\n') == 1
    assert json.loads(data.decode()[6:])['text'] == 'a\n\ndata: {bad}😀'
    with pytest.raises(ValueError):
        encode_event({'type': 'internal'})


def test_graph_consumer_close_cancels_inflight_and_releases_semaphore():
    async def run():
        sem = asyncio.Semaphore(1)
        closed = asyncio.Event()
        class Graph:
            async def astream(self, *args, **kw):
                async with sem:
                    try:
                        yield 'custom', {'type': 'progress', 'stage': 'generating'}
                        await asyncio.Future()
                    finally:
                        closed.set()
        stream = graph_events(Graph(), {}, {}, None)
        assert (await anext(stream))['type'] == 'progress'
        await stream.aclose()
        assert closed.is_set() and not sem.locked()
    asyncio.run(run())


def test_request_disconnect_cancels_whole_graph():
    async def run():
        started, closed = asyncio.Event(), asyncio.Event()
        class Request:
            async def is_disconnected(self):
                return started.is_set()
        class Graph:
            async def astream(self, *args, **kwargs):
                try:
                    started.set()
                    await asyncio.Future()
                    yield 'custom', {'type': 'token', 'text': 'not reached'}
                finally:
                    closed.set()
        async def consume():
            return [e async for e in graph_events(Graph(), {}, {}, None, request=Request())]
        assert await asyncio.wait_for(consume(), .5) == []
        assert closed.is_set()
    asyncio.run(run())


def test_session_locks_serial_and_reclaimed():
    async def run():
        locks, active, maximum = SessionLocks(), 0, 0
        async def work():
            nonlocal active, maximum
            async with locks.hold('same'):
                active += 1
                maximum = max(active, maximum)
                await asyncio.sleep(.002)
                active -= 1
        await asyncio.gather(*(work() for _ in range(10)))
        assert maximum == 1 and not locks.entries
    asyncio.run(run())


def test_disconnect_real_graph_stream_releases_client_and_keeps_history_empty(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessageChunk
    from ragagent.graph.deps import Deps
    from ragagent.preprocess.schema import Preprocessed

    async def run():
        cfg = config(tmp_path)
        started, closed, disconnected = asyncio.Event(), asyncio.Event(), asyncio.Event()
        sem = asyncio.Semaphore(1)

        class StreamingClient:
            async def astream(self, messages):
                try:
                    started.set()
                    yield AIMessageChunk(content='合成首字')
                    await asyncio.Future()
                finally:
                    closed.set()

        class Request:
            async def is_disconnected(self):
                return disconnected.is_set()

        async def structured(scene, eps, schema, messages):
            return Preprocessed(intent='chat', standalone_question='合成取消问题'), scene.primary

        monkeypatch.setattr('ragagent.llm.stream.build_client', lambda ep, scene: StreamingClient())
        dep = Deps(cfg=cfg, count=len, structured_call=structured, sems={'small': sem})
        db = str(tmp_path / 'cancel.db')
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            graph = build_graph(saver)
            stream = graph_events(graph, initial_state('合成取消问题', None),
                                  {'configurable': {'thread_id': 'cancel'}}, dep, request=Request())
            wire = []
            async for event in stream:
                wire.append(event)
                if event['type'] == 'token':
                    disconnected.set()
            assert started.is_set() and closed.is_set() and not sem.locked()
            assert wire[-1] == {'type': 'token', 'text': '合成首字'}
            assert not any(e['type'] in ('final', 'error') for e in wire)
            state = await graph.aget_state({'configurable': {'thread_id': 'cancel'}})
            assert not state.values.get('messages')
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            saved = await build_graph(saver).aget_state({'configurable': {'thread_id': 'cancel'}})
            assert not saved.values.get('messages')
    asyncio.run(asyncio.wait_for(run(), 3))


def test_rerank_degraded_budget_empty_context_real_graph_uses_rag_empty(tmp_path, llm_url, embed_url):
    cfg = config(tmp_path, llm_url)
    cfg = cfg.model_copy(update={'context': cfg.context.model_copy(update={'budget_tokens': 1})})
    async def unavailable(req):
        return httpx.Response(503, json={'detail': 'unavailable'})
    dep = deps(cfg, embed_url, rerank_client=httpx.AsyncClient(base_url=embed_url, transport=httpx.MockTransport(unavailable)))
    preprocess_script(llm_url)
    with api_client(cfg, dep) as client:
        wire = events(client.post('/api/chat', json={'question': '合成无资料问题', 'session_id': 'empty'}))
        state = client.portal.call(client.app.state.graph.aget_state, {'configurable': {'thread_id': 'empty'}}).values
        assert state['gen_meta']['scene'] == 'rag_empty'
        assert state['retrieval']['degraded'] and state['context_blocks'] == []
        assert wire[-1]['type'] == 'final' and wire[-1]['sources'] == []
        assert not state['gen_meta']['post']['checked_citations']
        assert state['gen_meta']['post']['citation_support'] is None
        assert len(state['messages']) == 2


@pytest.mark.parametrize('tokenizer_fails', [False, True])
def test_production_factory_wires_counter_and_closes_milvus_on_startup_failure(tmp_path, monkeypatch, tokenizer_fails):
    import pymilvus
    import ragagent.ingest.chunking.tokenizer as tokenizer
    from ragagent.llm import observability
    previous = object()
    monkeypatch.setattr(observability, 'METRICS', previous)
    cfg = config(tmp_path)
    calls = []
    class Milvus:
        def __init__(self, **kwargs):
            calls.append(('milvus', kwargs['uri']))
        def close(self):
            calls.append(('close',))
    counter = lambda text: len(text)
    def get_counter(name):
        calls.append(('tokenizer', name))
        assert observability.get_metrics() is app.state.metrics
        if tokenizer_fails:
            raise RuntimeError('local tokenizer init failed')
        return counter
    monkeypatch.setattr(pymilvus, 'MilvusClient', Milvus)
    monkeypatch.setattr(tokenizer, 'get_counter', get_counter)
    app = create_app(cfg)
    if tokenizer_fails:
        with pytest.raises(RuntimeError, match='tokenizer init failed'):
            with TestClient(app):
                pass
    else:
        with TestClient(app) as client:
            assert app.state.deps.count is counter
            assert isinstance(app.state.deps.milvus, Milvus)
            assert client.get('/metrics').status_code == 200
    assert calls == [('milvus', cfg.services.milvus.uri), ('tokenizer', cfg.chunking.tokenizer), ('close',)]
    assert observability.get_metrics() is previous


def test_global_metrics_bound_for_lifespan_then_restored(tmp_path, monkeypatch):
    from ragagent.llm import observability
    from ragagent.graph.deps import Deps
    previous = object()
    monkeypatch.setattr(observability, 'METRICS', previous)
    cfg = config(tmp_path)
    dep = Deps(cfg=cfg, count=len)
    app = create_app(cfg, deps=dep)
    with TestClient(app):
        assert dep.metrics is app.state.metrics
        assert observability.get_metrics() is app.state.metrics
    assert observability.get_metrics() is previous


def test_global_metrics_restored_even_if_cleanup_raises(tmp_path, monkeypatch):
    import importlib
    from ragagent.llm import observability
    from ragagent.graph.deps import Deps
    module = importlib.import_module('ragagent.api.app')
    previous = object()
    monkeypatch.setattr(observability, 'METRICS', previous)
    async def close_fails():
        raise RuntimeError('synthetic cleanup failure')
    monkeypatch.setattr(module, 'close_clients', close_fails)
    cfg = config(tmp_path)
    app = create_app(cfg, deps=Deps(cfg=cfg, count=len))
    with pytest.raises(RuntimeError, match='synthetic cleanup failure'):
        with TestClient(app):
            assert observability.get_metrics() is app.state.metrics
    assert observability.get_metrics() is previous
