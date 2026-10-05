"""LangGraph 状态（总揽.md §4.1）。各字段存的是 model_dump() 后的 dict。"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    messages: Annotated[list, add_messages]  # 多轮对话历史
    question: str                  # 原始问题
    standalone_question: str       # 指代消解后的完整问题，重排用
    queries: list[str]             # 本轮检索用的 1~3 条查询
    tried_queries: list[str]       # 之前用过的查询，改写重试时避免重复
    intent: str                    # rag / chat / clarify
    prev_intent: str | None        # 上一轮的 intent，防止连续反问
    clarify_question: str | None   # intent=clarify 时返回给用户
    filters: dict | None           # 硬过滤：接口参数传入，始终生效
    soft_filters: dict | None      # 软过滤：从问题里识别，只在首次检索生效
    use_scope: bool                # 接口参数：本轮是否从问题里识别范围（软过滤），前端“取消限定”传 false
    use_soft: bool                 # 当前检索是否带软过滤
    scope_note: str | None         # 展示给用户的生效范围
    candidates: list[dict]         # 合并后送进重排的前 30 个
    hits: list[dict]               # 重排筛选后的 top 8
    expansions: dict[str, list[dict]]  # chunk_id -> 前后 chunk / 引导段落 / 图片等
    retrieval: dict                # top_score、degraded、各步耗时
    retry_count: int               # 改写重试次数
    context_blocks: list[dict]     # 上下文段落（ContextBlock），带引用编号和 chunk_ids
    attachments: list[dict]        # 附件图片，随回答返回给前端
    answer: str                    # 最终回答
    gen_meta: dict
