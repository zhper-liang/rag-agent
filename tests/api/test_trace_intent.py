"""真实图经过 API 时，用请求自己的 trace config 同步分类，监控失败不影响回答。"""
import importlib
import json
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from ragagent.monitor.langfuse import build_client

from tests.integration.helpers import KEYS, api_client, config, deps, events, preprocess_script

app_module = importlib.import_module('ragagent.api.app')


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in KEYS:
        monkeypatch.setenv(name, 'test')


def capture_hook(monkeypatch, *, outcome=True):
    configs, calls = [], []
    original = app_module.trace_config

    def trace_config(*args, **kwargs):
        result = original(*args, **kwargs)
        configs.append(result)
        return result

    def hook(run_config, *, intent):
        calls.append((run_config, intent))
        if outcome == 'raise':
            raise RuntimeError('secret exception payload')
        return outcome

    monkeypatch.setattr(app_module, 'trace_config', trace_config)
    monkeypatch.setattr(app_module, 'update_trace_intent', hook)
    return configs, calls


@pytest.mark.parametrize('intent', ['rag', 'chat', 'clarify'])
@pytest.mark.parametrize('outcome', [True, False, 'raise'])
def test_api_trace_classification_and_failure_isolation(
        tmp_path, llm_url, embed_url, monkeypatch, caplog, intent, outcome):
    cfg = config(tmp_path, llm_url)
    dep = deps(cfg, embed_url)
    configs, calls = capture_hook(monkeypatch, outcome=outcome)
    preprocess_script(llm_url, intent=intent)
    with api_client(cfg, dep) as client:
        wire = events(client.post('/api/chat', json={'question': '合成问题', 'session_id': intent}))
        saved = client.portal.call(client.app.state.graph.aget_state,
                                   {'configurable': {'thread_id': intent}})
        assert wire[-1]['type'] == 'final'
        assert not any(event['type'] == 'error' for event in wire)
        assert saved.values['intent'] == intent
        assert len(saved.values['messages']) == 2
        assert len(configs) == 1 and calls
        assert all(run_config is configs[0] and actual == intent for run_config, actual in calls)
        assert configs[0]['metadata']['message_id'] == wire[-1]['message_id']
        assert configs[0]['configurable']['thread_id'] == intent
        assert 'route="' + intent + '",status="ok"' in client.get('/metrics').text
    if outcome == 'raise':
        assert 'trace intent update failed_type=RuntimeError' in caplog.text
        assert 'secret exception payload' not in caplog.text


def test_same_session_new_request_uses_own_config_and_effective_intent(
        tmp_path, llm_url, embed_url, monkeypatch):
    cfg = config(tmp_path, llm_url)
    dep = deps(cfg, embed_url)
    configs, calls = capture_hook(monkeypatch)
    with api_client(cfg, dep) as client:
        for expected in ['clarify', 'rag']:
            preprocess_script(llm_url, intent='clarify')
            wire = events(client.post('/api/chat', json={'question': '继续', 'session_id': 'same'}))
            assert wire[-1]['type'] == 'final'
            assert configs[-1]['metadata']['message_id'] == wire[-1]['message_id']
            assert any(run_config is configs[-1] and intent == expected for run_config, intent in calls)
        assert configs[0] is not configs[1]
        assert configs[0]['metadata']['message_id'] != configs[1]['metadata']['message_id']
        assert all(intent == ('clarify' if run_config is configs[0] else 'rag')
                   for run_config, intent in calls)


def test_fallback_metric_route_keeps_rag_trace_intent(tmp_path, llm_url, embed_url, monkeypatch):
    cfg = config(tmp_path, llm_url)
    cfg = cfg.model_copy(update={'grade': cfg.grade.model_copy(update={'max_retries': 0})})
    reranker = httpx.AsyncClient(base_url=embed_url, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={'scores': [0.] * len(json.loads(request.content)['passages'])})))
    dep = deps(cfg, embed_url, rerank_client=reranker)
    configs, calls = capture_hook(monkeypatch)
    preprocess_script(llm_url, intent='rag')
    with api_client(cfg, dep) as client:
        wire = events(client.post('/api/chat', json={'question': '没有资料', 'session_id': 'fallback'}))
        saved = client.portal.call(client.app.state.graph.aget_state,
                                   {'configurable': {'thread_id': 'fallback'}})
        assert wire[-1]['type'] == 'final'
        assert saved.values['gen_meta']['scene'] == 'fallback'
        assert calls and all(run_config is configs[0] and intent == 'rag' for run_config, intent in calls)
        assert 'route="fallback",status="ok"' in client.get('/metrics').text


@pytest.mark.parametrize('intent', ['rag', 'chat', 'clarify'])
def test_api_real_masked_sdk_exports_classified_root(
        tmp_path, llm_url, embed_url, monkeypatch, intent):
    class CaptureExporter(SpanExporter):
        def __init__(self):
            self.spans = []

        def export(self, spans):
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    cfg = config(tmp_path, llm_url)
    settings = cfg.monitor.langfuse.model_copy(update={'enabled': True, 'mask': True})
    cfg = cfg.model_copy(update={'monitor': cfg.monitor.model_copy(update={'langfuse': settings})})
    monkeypatch.setenv(settings.public_key_env, 'pk-l3-' + uuid4().hex)
    monkeypatch.setenv(settings.secret_key_env, 'sk-local-test')
    exporter = CaptureExporter()
    trace_client = build_client(cfg, tracer_provider=TracerProvider(), span_exporter=exporter)
    assert trace_client is not None
    original = app_module.update_trace_intent
    outcomes = []

    def hook(run_config, *, intent):
        result = original(run_config, intent=intent)
        outcomes.append((intent, result))
        return result

    monkeypatch.setattr(app_module, 'update_trace_intent', hook)
    dep = deps(cfg, embed_url)
    preprocess_script(llm_url, intent=intent)
    try:
        with TestClient(app_module.create_app(cfg, deps=dep, trace_client=trace_client)) as client:
            try:
                wire = events(client.post('/api/chat', json={
                    'question': '不能上报的私密问题', 'session_id': 'masked'}))
                assert wire[-1]['type'] == 'final'
            finally:
                client.portal.call(dep.embed_client.aclose)
                client.portal.call(dep.rerank_client.aclose)
        trace_client.flush()
        assert (intent, True) in outcomes
        roots = [dict(span.attributes) for span in exporter.spans if span.name == 'LangGraph']
        assert len(roots) == 1
        assert roots[0]['langfuse.trace.metadata.intent'] == intent
        assert roots[0]['langfuse.observation.metadata.intent'] == intent
        assert roots[0]['langfuse.observation.metadata.message_id'] == '[masked:32]'
        root_span = next(span for span in exporter.spans if span.name == 'LangGraph')
        assert format(root_span.context.trace_id, '032x') == Langfuse.create_trace_id(seed=wire[-1]['message_id'])
        exported = json.dumps([dict(span.attributes) for span in exporter.spans], ensure_ascii=False)
        assert '不能上报的私密问题' not in exported
        assert '[masked:' in exported
    finally:
        trace_client.shutdown()
