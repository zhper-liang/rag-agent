"""预处理与改写的结构化输出（用户问题预处理.md §5.2、§9）。"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Scope(BaseModel):
    doc_titles: list[str] = Field(default_factory=list)
    file_types: list[str] = Field(default_factory=list)

    @field_validator("doc_titles", "file_types")
    @classmethod
    def _cap(cls, v: list[str]) -> list[str]:
        # 用截断而不是 max_length：超量时不让整次调用校验失败
        return [s.strip()[:100] for s in v if s.strip()][:5]


class Preprocessed(BaseModel):
    intent: Literal["rag", "chat", "clarify"] = "rag"
    standalone_question: str
    extra_queries: list[str] = Field(default_factory=list)
    scope: Scope = Field(default_factory=Scope)
    clarify_question: str | None = None

    @field_validator("standalone_question")
    @classmethod
    def _cap_question(cls, v: str) -> str:
        return v.strip()[:500]

    @field_validator("extra_queries")
    @classmethod
    def _cap_queries(cls, v: list[str]) -> list[str]:
        # 超长、超量直接截断，不因为这类问题让整次调用失败。
        # 这里多留几条，去重后再由 build_queries 截到 3 条，避免重复查询占掉名额
        return [q.strip()[:200] for q in v if q.strip()][:4]


class Rewrite(BaseModel):
    queries: list[str]

    @field_validator("queries")
    @classmethod
    def _cap(cls, v: list[str]) -> list[str]:
        return [q.strip()[:200] for q in v if q.strip()]
