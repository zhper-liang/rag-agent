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
