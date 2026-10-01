"""上下文结果与优先级实际预算写入state，客户端不入checkpoint。"""

import asyncio
from time import perf_counter

from langgraph.runtime import Runtime

from ragagent.context.build import build_context_details
from ragagent.graph.deps import Deps, emit
from ragagent.graph.state import RAGState


async def build_context(state: RAGState, runtime: Runtime[Deps]) -> dict:
    start = perf_counter()
    emit(runtime, {"type": "progress", "stage": "building_context"})
    try:
        blocks, images, usage = await asyncio.to_thread(build_context_details, state["hits"],
                                                      state["expansions"], runtime.context.cfg, metrics=runtime.context.metrics)
    finally:
        if runtime.context.metrics is not None:
            runtime.context.metrics.stage_seconds.labels(stage="build_context").observe(perf_counter() - start)
    retrieval = {**state.get("retrieval", {}), "context_usage": usage,
                 "timings_ms": {**state.get("retrieval", {}).get("timings_ms", {}),
                                "build_context": round((perf_counter() - start) * 1000)}}
    # ImageRef路径由API按约定映射为安全URL；此处不猜前端base_url。
    emit(runtime, {"type": "meta", "scope_note": state.get("scope_note"),
                   "attachments": [i.model_dump() for i in images]})
    return {"context_blocks": [b.model_dump() for b in blocks], "attachments": [i.model_dump() for i in images],
            "retrieval": retrieval}
