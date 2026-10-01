"""条件边无副作用；阈值由 Runtime context 注入，不读取全局配置。"""

from langgraph.runtime import Runtime

from ragagent.graph.deps import Deps
from ragagent.graph.state import RAGState


def route_after_preprocess(state: RAGState) -> str:
    return {"rag": "retrieve", "chat": "generate", "clarify": "clarify"}[state["intent"]]


def route_after_rerank(state: RAGState, runtime: Runtime[Deps]) -> str:
    cfg = runtime.context.cfg.grade
    if state["retrieval"].get("degraded") or state["retrieval"].get("top_score", 0) >= cfg.pass_score:
        return "build_context"
    return "rewrite" if state.get("retry_count", 0) < cfg.max_retries else "fallback"


def route_after_rewrite(state: RAGState) -> str:
    return "retrieve" if state["queries"] else "fallback"
