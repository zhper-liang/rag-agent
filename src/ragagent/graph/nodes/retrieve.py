"""有效软硬过滤→编码→hybrid→合并→回表。"""

from time import perf_counter

from langgraph.runtime import Runtime

from ragagent.graph.deps import Deps, emit
from ragagent.graph.state import RAGState
from ragagent.preprocess.filters import effective_filters
from ragagent.retrieval.hybrid import retrieve as run_retrieve


async def retrieve(state: RAGState, runtime: Runtime[Deps]) -> dict:
    start = perf_counter()
    emit(runtime, {"type": "progress", "stage": "retrieving"})
    filters = effective_filters(state.get("filters"), state.get("soft_filters") if state.get("use_soft", True) else None)
    try:
        candidates, result = await run_retrieve(state["queries"], filters, runtime.context)
    finally:
        if runtime.context.metrics is not None:
            runtime.context.metrics.stage_seconds.labels(stage="retrieve").observe(perf_counter() - start)
    old = state.get("retrieval", {})
    retrieval = {**old, **result.model_dump(), "timings_ms": {**old.get("timings_ms", {}), **result.timings_ms}}
    retrieval["scope_doc_ids"] = (filters or {}).get("doc_ids", [])
    return {"candidates": [h.model_dump() for h in candidates], "retrieval": retrieval}
