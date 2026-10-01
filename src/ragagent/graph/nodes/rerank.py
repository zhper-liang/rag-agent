"""重排→筛选→批量扩展，grade在条件边执行。"""

import asyncio
from time import perf_counter

from langgraph.runtime import Runtime

from ragagent.graph.deps import Deps, emit
from ragagent.graph.state import RAGState
from ragagent.retrieval.expand import expand
from ragagent.retrieval.rerank import rerank as run_rerank


async def rerank(state: RAGState, runtime: Runtime[Deps]) -> dict:
    deps = runtime.context
    start = perf_counter()
    emit(runtime, {"type": "progress", "stage": "reranking"})
    try:
        hits, result = await run_rerank(state["standalone_question"], state["candidates"], deps, state.get("retrieval"))
        exp_start = perf_counter()
        expansions = await asyncio.to_thread(expand, deps.milvus, deps.cfg.services.milvus.collection, hits, deps.cfg)
        result.timings_ms["expand"] = round((perf_counter() - exp_start) * 1000)
    finally:
        if runtime.context.metrics is not None:
            runtime.context.metrics.stage_seconds.labels(stage="rerank").observe(perf_counter() - start)
    old = state.get("retrieval", {})
    retrieval = {**old, **result.model_dump(), "timings_ms": {**old.get("timings_ms", {}), **result.timings_ms}}
    return {"hits": [h.model_dump() for h in hits],
            "expansions": {cid: [c.model_dump() for c in values] for cid, values in expansions.items()},
            "retrieval": retrieval}
