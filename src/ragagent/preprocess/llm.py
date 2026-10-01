"""预处理依赖 L3 structured_call 的主备逻辑，不另行重试（§5/10）。"""

import logging

from ragagent.monitor.metrics import DEGRADED
from ragagent.preprocess.history import history_turns, truncate_history
from ragagent.preprocess.normalize import normalize_question
from ragagent.preprocess.queries import build_queries, dedup_queries
from ragagent.preprocess.schema import Preprocessed
from ragagent.preprocess.scope import resolve_scope

log = logging.getLogger(__name__)


async def call_structured(name, schema, variables, deps):
    messages = [{"role": "system", "content": deps.prompt_env.get_template(f"{name}_system.j2").render()},
                {"role": "user", "content": deps.prompt_env.get_template(f"{name}_user.j2").render(**variables)}]
    try:
        result, endpoint = await deps.structured_call(getattr(deps.cfg.llm.scenes, name),
                                                    deps.cfg.llm.endpoints, schema, messages)
        if result is None:
            return None
        return schema.model_validate(result)
    except Exception as exc:
        log.warning("%s failed_type=%s", name, type(exc).__name__)
        return None


async def preprocess_question(question, messages, prev_intent, deps, filters=None):
    question = normalize_question(question, deps.cfg.preprocess.max_question_chars)
    history = truncate_history(messages, deps.cfg.preprocess.history, deps.count)
    parsed = await call_structured("preprocess", Preprocessed, {"question": question, "history": history}, deps)
    degraded = parsed is None or not parsed.standalone_question.strip()
    if degraded:
        DEGRADED.labels(kind="preprocess").inc()
        log.warning("preprocess degraded; use original query")
        turns = history_turns(messages)
        queries = [question]
        if turns:
            queries.append(turns[-1]["question"] + " " + question)
        return {"intent": "rag", "standalone_question": question, "queries": dedup_queries(queries),
                "soft_filters": None, "scope_note": None, "clarify_question": None,
                "preprocess_degraded": True, "scope_doc_ids": []}
    intent = "rag" if parsed.intent == "clarify" and prev_intent == "clarify" else parsed.intent
    soft, note, scope_ids = resolve_scope(parsed.scope, deps.registry, deps.cfg.preprocess.scope, filters)
    return {"intent": intent, "standalone_question": parsed.standalone_question,
            "queries": build_queries(parsed), "soft_filters": soft, "scope_note": note,
            "clarify_question": (parsed.clarify_question or "请补充您想了解的具体问题") if intent == "clarify" else None,
            "preprocess_degraded": False, "scope_doc_ids": scope_ids}
