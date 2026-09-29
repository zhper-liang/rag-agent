"""检索结果：Hit、Expansion、RetrievalResult（检索与重排.md §4.5）。"""

from typing import Literal

from pydantic import BaseModel, Field

from ragagent.index.models import ChunkFields


class Hit(ChunkFields):
    score: float                        # 融合分数（WeightedRanker），多查询取最高
    rerank_score: float | None = None


ExpansionRole = Literal["section", "neighbor", "lead_in", "split_sibling", "attachment", "anchor"]


class Expansion(ChunkFields):
    role: ExpansionRole


class RetrievalResult(BaseModel):
    """RAGState.retrieval 的结构。"""
    top_score: float = 0.0              # 重排最高分，被 per_doc_cap 挡掉的也算；没有候选时为 0
    degraded: bool = False              # 重排超时或报错（§8），grade 直接放行
    dense_only: bool = False            # hybrid_search 失败后只查了稠密向量（§8）
    timings_ms: dict[str, int] = Field(default_factory=dict)
