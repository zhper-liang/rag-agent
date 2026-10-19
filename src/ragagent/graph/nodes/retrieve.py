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
    time_ids = runtime.context.time_doc_ids
    if time_ids is not None and (filters or {}).get("doc_ids") and not set(filters["doc_ids"]) & set(time_ids):
        # 与时间硬范围冲突时，丢弃整套软范围，而不是悄悄放宽硬过滤。
        filters = effective_filters(state.get("filters"), None)
    try:
        candidates, result = await run_retrieve(state["queries"], filters, runtime.context)
    finally:
        if runtime.context.metrics is not None:
            runtime.context.metrics.stage_seconds.labels(stage="retrieve").observe(perf_counter() - start)
    old = state.get("retrieval", {})
    retrieval = {**old, **result.model_dump(), "timings_ms": {**old.get("timings_ms", {}), **result.timings_ms}}
    ids = (filters or {}).get("doc_ids", [])
    retrieval["scope_doc_ids"] = sorted(set(ids) & set(time_ids)) if time_ids is not None and ids else (time_ids or ids)
    return {"candidates": [h.model_dump() for h in candidates], "retrieval": retrieval}
