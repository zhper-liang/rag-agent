"""一次只放宽范围或改查询；无新查询就让路由进入 fallback（§9）。"""

from ragagent.preprocess.llm import call_structured
from ragagent.preprocess.queries import dedup_queries
from ragagent.preprocess.schema import Rewrite


async def rewrite_queries(state, deps):
    queries = list(state.get("queries", []))
    tried = dedup_queries([*state.get("tried_queries", []), *queries], limit=10**9)
    common = {"retry_count": state.get("retry_count", 0) + 1, "tried_queries": tried}
    if state.get("use_soft", True) and state.get("soft_filters"):
        return {**common, "use_soft": False, "queries": queries,
                "scope_note": "限定范围内没有找到，已扩大到全部文档"}
    hints = [f"{c['title']} > {c['section_path']}" for c in state.get("candidates", [])[:deps.cfg.rewrite.hint_candidates]]
    parsed = await call_structured("rewrite", Rewrite,
                                   {"question": state["standalone_question"], "tried": tried, "hints": hints}, deps)
    new = dedup_queries(parsed.queries, limit=deps.cfg.rewrite.max_queries, tried=tried) if parsed else []
    return {**common, "queries": new}
