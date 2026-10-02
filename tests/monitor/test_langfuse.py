import json
import uuid
from types import SimpleNamespace

from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from ragagent.monitor.langfuse import KEEP, _mask, build_client, get_handler, mask, report_score


def config(enabled=True, masked=True):
    return SimpleNamespace(monitor=SimpleNamespace(langfuse=SimpleNamespace(
        enabled=enabled, mask=masked, host="http://127.0.0.1:18499",
        public_key_env="L4_TEST_PUBLIC", secret_key_env="L4_TEST_SECRET")))


def test_recursive_mask_exact_whitelist_lists_models_and_no_mutation():
    data = {"question": "私密问题", "chunk_id": "abc", "unknown": ["secret", {"text": "正文", "model": "m"}],
            "chunks": [{"doc_id": "id", "version": 2, "text": "机密"}], "seq": 3,
            "flag": True, "score": .5, "null": None, "intent": ["rag", "chat"]}
    out = mask(data=data, ignored="unused")
    assert out == {"question": "[masked:4]", "chunk_id": "abc", "unknown": ["[masked:6]", {"text": "[masked:2]", "model": "m"}],
                   "chunks": [{"doc_id": "id", "version": 2, "text": "[masked:2]"}], "seq": 3,
                   "flag": True, "score": .5, "null": None, "intent": ["rag", "chat"]}
    assert data["question"] == "私密问题"
    for key in KEEP:
        assert _mask({key: "keep", "private": "hide"}) == {key: "keep", "private": "[masked:4]"}
    assert _mask(b"secret") == "[masked:6]"


class CaptureExporter(SpanExporter):
    def __init__(self):
        self.spans = []
    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS
    def shutdown(self):
        pass


def test_real_sdk_export_and_callback_mask_no_network():
    exporter = CaptureExporter()
    env = {"L4_TEST_PUBLIC": "pk-l4-" + uuid.uuid4().hex, "L4_TEST_SECRET": "sk-test"}
    client = build_client(config(), env=env, tracer_provider=TracerProvider(), span_exporter=exporter)
    assert client is not None
    with client.start_as_current_observation(as_type="span", name="test", input={"question": "秘密问题", "intent": "rag"}) as span:
        span.update(output={"answer": "资料原文"}, metadata={"chunk_id": "abc", "body": "私密片段"})
    handler = get_handler(config(), env=env, client=client, message_id="message-1")
    assert handler is not None
    assert handler._langfuse_client._mask is mask
    handler.on_chain_start({"name": "chain"}, {"question": "callback私密"}, run_id=uuid.UUID(int=1))
    handler.on_chain_end({"answer": "callback正文"}, run_id=uuid.UUID(int=1))
    client.flush()
    exported = json.dumps([dict(s.attributes or {}) for s in exporter.spans], ensure_ascii=False)
    assert exporter.spans
    for secret in ("秘密问题", "资料原文", "私密片段", "callback私密", "callback正文"):
        assert secret not in exported
    assert "masked" in exported and "abc" in exported
    client.shutdown()


def test_disabled_missing_keys_unmasked_are_fail_closed_and_score_only():
    assert build_client(config(enabled=False), env={}) is None
    assert build_client(config(), env={}) is None
    assert build_client(config(masked=False), env={}) is None
    calls = []
    class ScoreClient:
        def create_score(self, **kwargs):
            calls.append(kwargs)
    report_score(ScoreClient(), message_id="m", rating=-1)
    assert calls == [{"name": "feedback", "value": -1, "data_type": "NUMERIC", "trace_id": Langfuse.create_trace_id(seed="m")}]


def test_real_sdk_root_intent_update_and_final_state_isolate_traces(caplog):
    from ragagent.monitor.langfuse import trace_config, update_trace_intent
    exporter = CaptureExporter()
    env = {'L4_TEST_PUBLIC': 'pk-l4-' + uuid.uuid4().hex, 'L4_TEST_SECRET': 'sk-test'}
    cfg = config()
    cfg.prompt = SimpleNamespace(version='v1')
    client = build_client(cfg, env=env, tracer_provider=TracerProvider(), span_exporter=exporter)
    configs = [trace_config(cfg, session_id='s', message_id=f'm{i}', config_hash='hash',
                            intent='rag', client=client, env=env) for i in range(2)]
    for i, (run_config, final) in enumerate(zip(configs, ['chat', 'clarify'])):
        handler = run_config['callbacks'][0]
        rid = uuid.uuid4()
        handler.on_chain_start({'name': 'LangGraph'}, {'question': '禁止上报原文'}, run_id=rid,
                               metadata=run_config['metadata'])
        if i == 0:
            assert update_trace_intent(run_config, intent=final)
            assert not update_trace_intent(run_config, intent='私密意图')
        handler.on_chain_end({'intent': final, 'answer': '禁止答案原文'}, run_id=rid)
        assert not update_trace_intent(run_config, intent='rag')
    client.flush()
    spans = [dict(s.attributes) for s in exporter.spans if s.name == 'LangGraph']
    assert len(spans) == 2
    for attrs, final in zip(spans, ['chat', 'clarify']):
        assert attrs['langfuse.trace.metadata.intent'] == final
        assert attrs['langfuse.observation.metadata.intent'] == final
        assert 'langfuse.observation.metadata.message_id' in attrs
    exported = json.dumps(spans, ensure_ascii=False)
    assert '禁止上报原文' not in exported and '禁止答案原文' not in exported
    handler = configs[0]['callbacks'][0]
    class BrokenRoot:
        class Span:
            def is_recording(self):
                return True
        _otel_span = Span()
        def update(self, **kwargs):
            raise RuntimeError('不能出现在日志的失败内容')
    handler._safe_root = BrokenRoot()
    assert not update_trace_intent(configs[0], intent='chat')
    assert 'RuntimeError' in caplog.text and '不能出现在日志' not in caplog.text
    assert not update_trace_intent({'callbacks': []}, intent='chat')
    client.shutdown()


def test_proxy_transport_exports_masked_v4_payload_to_http(monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.responses import Response
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    from tests.stubs.serve import serve
    app = FastAPI()
    captured = []
    @app.post('/api/public/otel/v1/traces')
    async def receive(request: Request):
        captured.append((dict(request.headers), await request.body()))
        return Response(b'', media_type='application/x-protobuf')
    monkeypatch.setenv('HTTPS_PROXY', 'http://127.0.0.1:1')
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')
    with serve(app) as host:
        cfg = config()
        cfg.monitor.langfuse.host = host
        env = {'L4_TEST_PUBLIC':'pk-'+uuid.uuid4().hex, 'L4_TEST_SECRET':'sk-test',
               'HTTPS_PROXY':'http://127.0.0.1:1'}
        client = build_client(cfg, env=env, tracer_provider=TracerProvider(), timeout=2)
        with client.start_as_current_observation(as_type='span', name='proxy-test',
                                               input={'question':'不能传输的秘密'}) as span:
            span.update(output={'answer':'不能传输的答案'}, metadata={'intent':'chat'})
        client.flush()
        client.shutdown()
    assert len(captured) == 1
    headers, payload = captured[0]
    assert headers['x-langfuse-ingestion-version'] == '4'
    message = ExportTraceServiceRequest.FromString(payload)
    text = str(message)
    assert '不能传输的秘密' not in text and '不能传输的答案' not in text
    assert 'masked' in text and 'chat' in text
