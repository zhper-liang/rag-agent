"""每轮初始化必须清空所有单轮字段；messages和硬过滤不清空（§13）。"""

from time import perf_counter

from langgraph.runtime import Runtime

from ragagent.graph.deps import Deps, emit
from ragagent.graph.state import RAGState
from ragagent.preprocess.llm import preprocess_question
from ragagent.preprocess.normalize import QuestionError
from ragagent.retrieval.filters import validate_filters
from ragagent.retrieval.models import RetrievalResult


def reset_turn(state: RAGState) -> dict:
    prev_intent = state.get("intent")
    hard = validate_filters(state.get("filters")) or None
    return {"prev_intent": prev_intent, "retry_count": 0, "tried_queries": [], "use_soft": True,
             "filters": hard, "soft_filters": None, "scope_note": None, "clarify_question": None,
             "candidates": [], "hits": [], "expansions": {}, "context_blocks": [], "attachments": [],
             "answer": "", "gen_meta": {}, "retrieval": RetrievalResult().model_dump()}


async def preprocess(state: RAGState, runtime: Runtime[Deps]) -> dict:
    start = perf_counter()
    emit(runtime, {"type": "progress", "stage": "preprocessing"})
    reset = reset_turn(state)
    prev_intent, hard = reset["prev_intent"], reset["filters"]
    try:
        parsed = await preprocess_question(state.get("question", ""), state.get("messages", []),
                                           prev_intent, runtime.context, hard)
    except QuestionError as exc:
        parsed = {"intent": "clarify", "standalone_question": "", "queries": [],
                  "soft_filters": None, "scope_note": None, "clarify_question": str(exc),
                  "preprocess_degraded": False, "scope_doc_ids": []}
    elapsed = perf_counter() - start
    if runtime.context.metrics is not None:
        runtime.context.metrics.stage_seconds.labels(stage="preprocess").observe(elapsed)
    degraded, scope_ids = parsed.pop("preprocess_degraded"), parsed.pop("scope_doc_ids")
    retrieval = reset["retrieval"]
    retrieval["timings_ms"]["preprocess"] = round(elapsed * 1000)
    retrieval["preprocess"] = {"degraded": degraded, "intent": parsed["intent"]}
    retrieval["scope_doc_ids"] = scope_ids
    return {**reset, **parsed}
