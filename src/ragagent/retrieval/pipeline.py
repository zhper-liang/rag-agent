"""快速档：仅单轮前半链路，不生成、不循环grade/rewrite、不装Langfuse。"""

from time import perf_counter

from langgraph.runtime import Runtime

from ragagent.graph.nodes.build_context import build_context
from ragagent.graph.nodes.preprocess import preprocess, reset_turn
from ragagent.graph.nodes.rerank import rerank
from ragagent.graph.nodes.retrieve import retrieve
from ragagent.graph.nodes.routes import route_after_preprocess, route_after_rerank
from ragagent.preprocess.history import previous_intent
from ragagent.preprocess.normalize import QuestionError, normalize_question


async def run_pipeline(question: str, deps, *, filters=None, messages=None) -> dict:
    start = perf_counter()
    # Runtime显式注入与图一致；无stream消费者时事件只被丢弃，不进State。
    runtime = Runtime(context=deps, stream_writer=lambda event: None)
    state = {"question": question, "filters": filters, "messages": list(messages or []),
             "intent": previous_intent(messages)}
    state.update(await preprocess(state, runtime))
    return await _finish(state, runtime, start)


async def _finish(state, runtime, start):
    route = route_after_preprocess(state)
    if route == "retrieve":
        state.update(await retrieve(state, runtime))
        state.update(await rerank(state, runtime))
        route = route_after_rerank(state, runtime)
        # 即使grade未过也构造上下文，供quick单轮召回评测；不生成或发起重试。
        state.update(await build_context(state, runtime))
    state["retrieval"]["route"] = route
    state["retrieval"]["grade"] = "pass" if route == "build_context" else ("fail" if route in {"rewrite", "fallback"} else "not_applicable")
    state["retrieval"]["total_ms"] = round((perf_counter() - start) * 1000)
    state["retrieval"]["mode"] = "quick_single_round"
    return state


async def run_original(question: str, deps, *, filters=None, messages=None) -> dict:
    """原问题对照：仅规则规范化+hard过滤，不LLM、不scope、不指代消解。"""
    start = perf_counter()
    runtime = Runtime(context=deps, stream_writer=lambda event: None)
    state = {"question": question, "filters": filters, "messages": list(messages or []),
             "intent": previous_intent(messages)}
    state.update(reset_turn(state))
    try:
        normalized = normalize_question(question, deps.cfg.preprocess.max_question_chars)
        state.update(intent="rag", standalone_question=normalized, queries=[normalized])
    except QuestionError as exc:
        state.update(intent="clarify", standalone_question="", queries=[], clarify_question=str(exc))
    state["retrieval"]["preprocess"] = {"degraded": False, "intent": state["intent"], "skipped": True}
    state["retrieval"]["scope_doc_ids"] = []
    state["retrieval"]["timings_ms"]["preprocess"] = round((perf_counter() - start) * 1000)
    return await _finish(state, runtime, start)
