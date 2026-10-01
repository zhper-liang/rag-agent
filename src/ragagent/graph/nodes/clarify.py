"""非LLM回答整段发一次token，随后由L3 postprocess保存本轮。"""

from langgraph.runtime import Runtime

from ragagent.graph.deps import Deps, emit
from ragagent.graph.state import RAGState


async def clarify(state: RAGState, runtime: Runtime[Deps]) -> dict:
    answer = state.get("clarify_question") or "请补充您想了解的具体问题"
    emit(runtime, {"type": "token", "text": answer})
    return {"answer": answer, "gen_meta": {"status": "ok", "scene": "clarify"}}
