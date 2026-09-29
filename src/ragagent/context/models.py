"""上下文段落：ImageRef、ContextBlock（上下文构造.md §8）。"""

from pydantic import BaseModel, Field


class ImageRef(BaseModel):
    chunk_id: str
    image_path: str
    caption: str
    send_original: bool


class ContextBlock(BaseModel):
    cite: int                  # 引用编号 [1] [2] ...
    doc_id: str
    title: str
    section_path: str          # 段落第一个 chunk 的标题路径
    page_start: int            # 没有页码时为 -1
    page_end: int
    text: str                  # 合并、去重叠、插入标题后的 Markdown
    chunk_ids: list[str]       # 组成这个段落的全部 chunk，后处理校验引用用
    hit_ids: list[str]         # 其中哪些是命中（其余是扩展内容）
    score: float               # 段落内最高的重排分数
    images: list[ImageRef] = Field(default_factory=list)
