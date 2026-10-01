"""重试计数和tried_queries在此推进，不持有上轮检索结果。"""

from langgraph.runtime import Runtime

from ragagent.graph.deps import Deps, emit
from ragagent.graph.state import RAGState
from ragagent.preprocess.rewrite import rewrite_queries


async def rewrite(state: RAGState, runtime: Runtime[Deps]) -> dict:
    emit(runtime, {"type": "progress", "stage": "rewriting"})
    if runtime.context.metrics is not None:
        runtime.context.metrics.retries.inc()
    return await rewrite_queries(state, runtime.context)
