"""§10 指标对象及严格标签白名单；不允许会话/用户/任意异常内容作标签。"""
from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

ROUTES = {"rag", "chat", "clarify", "fallback"}
STAGES = {"preprocess", "retrieve", "rerank", "build_context", "ttft", "generate", "postprocess"}
SCENES = {"generate", "chat", "preprocess", "rewrite", "eval"}
ENDPOINTS = {"main", "main_backup", "small", "small_backup"}


class FixedLabels:
    """保持 prometheus-client 的 labels/inc/observe/time 用法，先验证再建序列。"""
    def __init__(self, metric, allowed: dict[str, set[str]]):
        self._metric, self.allowed = metric, allowed

    def labels(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError("use positional or keyword labels, not both")
        if args:
            if len(args) != len(self.allowed):
                raise ValueError("incorrect label count")
            kwargs = dict(zip(self.allowed, args))
        if set(kwargs) != set(self.allowed):
            raise ValueError("incorrect label names")
        normalized = {k: str(v) for k, v in kwargs.items()}
        if any(v not in self.allowed[k] for k, v in normalized.items()):
            raise ValueError("unapproved metric label value")
        return self._metric.labels(**normalized)

    def inc(self, amount=1):
        return self._metric.inc(amount)


class MonitorMetrics:
    def __init__(self, *, registry=None, endpoints=None):
        self.registry = registry if registry is not None else CollectorRegistry()
        endpoints = set(endpoints) if endpoints is not None else ENDPOINTS

        def counter(name, labels):
            return FixedLabels(Counter(name, name, list(labels), registry=self.registry), labels)

        self.requests = counter("rag_requests_total", {"route": ROUTES, "status": {"ok", "error"}})
        self.stage_seconds = FixedLabels(Histogram("rag_stage_seconds", "Stage duration in seconds", ["stage"],
                                                   registry=self.registry), {"stage": STAGES})
        self.llm_calls = counter("rag_llm_calls_total", {"scene": SCENES, "endpoint": endpoints,
                                                       "outcome": {"ok", "retry", "fallback", "failed"}})
        self.llm_tokens = counter("rag_llm_tokens_total", {"scene": SCENES, "endpoint": endpoints,
                                                         "kind": {"input", "output"}})
        self.degraded = counter("rag_degraded_total", {"kind": {"preprocess", "rerank", "hybrid", "verify", "image"}})
        self.retries = counter("rag_retries_total", {})
        self.fallback = counter("rag_fallback_total", {})
        self.citations = counter("rag_citations_total", {"state": {"valid", "invalid", "supported", "unsupported", "covered", "uncovered"}})
        self.feedback = counter("rag_feedback_total", {"rating": {"1", "-1"}})

    def router(self, path="/metrics") -> APIRouter:
        router = APIRouter()

        @router.get(path, include_in_schema=False)
        def metrics():
            return Response(generate_latest(self.registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

        return router


METRICS = MonitorMetrics()
REQUESTS = METRICS.requests
STAGE_SECONDS = METRICS.stage_seconds
LLM_CALLS = METRICS.llm_calls
LLM_TOKENS = METRICS.llm_tokens
DEGRADED = METRICS.degraded
RETRIES = METRICS.retries
FALLBACK = METRICS.fallback
CITATIONS = METRICS.citations
FEEDBACK = METRICS.feedback
