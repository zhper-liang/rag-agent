import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from ragagent.monitor.feedback import FeedbackInput, FeedbackStore, create_feedback_router, export_downvotes
from ragagent.monitor.metrics import MonitorMetrics
from scripts.gen_alerts import generate_rules


def cfg(tmp_path, limit=2):
    return SimpleNamespace(paths=SimpleNamespace(feedback_db=str(tmp_path / "feedback.db")),
                           monitor=SimpleNamespace(feedback=SimpleNamespace(reason_max_chars=5, rate_limit_per_session_min=limit)))


def test_metrics_types_fixed_labels_and_http_endpoint():
    metrics = MonitorMetrics()
    metrics.requests.labels(route="rag", status="ok").inc()
    metrics.stage_seconds.labels(stage="ttft").observe(.2)
    metrics.llm_calls.labels("eval", "main", "ok").inc()
    metrics.llm_tokens.labels("eval", "main", "output").inc(10)
    metrics.degraded.labels("verify").inc()
    metrics.retries.inc()
    metrics.fallback.inc()
    metrics.citations.labels("supported").inc()
    metrics.feedback.labels("-1").inc()
    with pytest.raises(ValueError):
        metrics.requests.labels(route="session-123", status="ok")
    with pytest.raises(ValueError):
        metrics.llm_calls.labels(scene="eval", endpoint="user-supplied", outcome="ok")
    with pytest.raises(ValueError):
        metrics.feedback.labels(rating="reason-secret")
    app = FastAPI()
    app.include_router(metrics.router())
    response = TestClient(app).get("/metrics")
    assert response.status_code == 200
    text = response.text
    for name in ("rag_requests", "rag_llm_calls", "rag_llm_tokens", "rag_degraded", "rag_retries", "rag_fallback", "rag_citations", "rag_feedback"):
        assert f"# TYPE {name}_total counter" in text
    assert "# TYPE rag_stage_seconds histogram" in text
    assert 'rag_stage_seconds_count{stage="ttft"} 1.0' in text
    assert "session-123" not in text


def test_feedback_membership_limits_reason_privacy_and_score_failure(tmp_path):
    config, metrics, scores = cfg(tmp_path), MonitorMetrics(), []
    history = {"a": [{"content": "问题"}, {"content": "答案", "additional_kwargs": {"message_id": "m", "post": {"support": .8}, "timings": {"generate": 2}}}], "b": []}

    async def lookup(sid):
        return history.get(sid, [])

    def score(*args):
        scores.append(args)
        raise RuntimeError("cloud unavailable")

    store = FeedbackStore(config.paths.feedback_db)
    app = FastAPI()
    app.include_router(create_feedback_router(config, lookup, metrics=metrics, score_reporter=score, store=store))
    client = TestClient(app)
    body = {"message_id": "m", "rating": -1, "reason": "私密理由"}
    assert client.post("/api/feedback", json=body).status_code == 400
    assert client.post("/api/feedback", headers={"X-Session-ID": "b"}, json=body).status_code == 404
    assert client.post("/api/feedback", headers={"X-Session-ID": "a"}, json={**body, "reason": "x" * 6}).status_code == 422
    for rating in (0, True, "1"):
        assert client.post("/api/feedback", headers={"X-Session-ID": "a"}, json={**body, "rating": rating}).status_code == 422
    assert client.post("/api/feedback", headers={"X-Session-ID": "a"}, json=body).status_code == 200
    assert client.post("/api/feedback", headers={"X-Session-ID": "a"}, json=body).status_code == 200
    assert client.post("/api/feedback", headers={"X-Session-ID": "a"}, json=body).status_code == 429
    assert scores == [("a", "m", -1), ("a", "m", -1)]  # 不传 reason
    assert store.downvotes()[0]["reason"] == "私密理由"
    exported = asyncio.run(export_downvotes(store, lookup))
    assert exported[0]["question"] == "问题"
    assert exported[0]["answer"] == "答案"
    assert exported[0]["citations"] == {"support": .8}
    assert b"rag_feedback_total" in generate_latest(metrics.registry)


def test_feedback_rate_limit_atomic_and_expires(tmp_path):
    now = [100.0]
    store = FeedbackStore(tmp_path / "f.db", clock=lambda: now[0])
    data = FeedbackInput(message_id="m", rating=1)
    def save(_):
        try:
            store.save("s", data, 3)
            return 200
        except HTTPException as e:
            return e.status_code
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(save, range(8)))
    assert results.count(200) == 3
    assert results.count(429) == 5
    now[0] += 60
    assert save(0) == 200


def test_alert_values_and_durations_come_only_from_config():
    original = yaml.safe_load(open("config.yaml"))["monitor"]["alerts"]
    alerts = {k: {**v, "value": .4321 if "ratio" in k else 12.345,
                  **({"for": "37m"} if "for" in v else {}),
                  **({"window": "2h", "min_count": 73} if "window" in v else {})} for k, v in original.items()}
    doc = generate_rules(alerts)
    parsed = yaml.safe_load(yaml.safe_dump(doc))
    assert parsed == doc
    assert len(doc["groups"][0]["rules"]) == 7
    text = yaml.safe_dump(doc)
    assert "0.4321" in text and "12.345" in text and "37m" in text and "2h" in text and "73" in text
    assert str(12.345 * 1024**3) in text
    with pytest.raises(ValueError):
        generate_rules({**original, "unknown": {"value": 1, "for": "1m"}})
    with pytest.raises(ValueError):
        generate_rules({**original, "error_ratio": {"value": .2, "for": '1m) or vector(1)'}})
