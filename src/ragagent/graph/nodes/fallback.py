from langgraph.runtime import Runtime

from ragagent.graph.deps import Deps, emit
from ragagent.graph.state import RAGState
from ragagent.prompt.fallback import render_fallback


async def fallback_node(state: RAGState, runtime: Runtime[Deps]) -> dict:
    answer = render_fallback(state, runtime.context.cfg, env=runtime.context.prompt_env)
    emit(runtime, {'type': 'token', 'text': answer})
    if runtime.context.metrics is not None:
        runtime.context.metrics.fallback.inc()
    return {'answer': answer, 'gen_meta': {'status': 'ok', 'scene': 'fallback'}}
