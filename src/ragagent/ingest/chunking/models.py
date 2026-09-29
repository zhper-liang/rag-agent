"""切分的数据结构：Chunk、Draft、Section、Unit（文档切分.md §4~§6）。"""

import re
from typing import Literal

from pydantic import BaseModel, Field

from ragagent.ingest.models import DOC_ID_PATTERN, Block

ChunkType = Literal["text", "code", "image"]

CHUNK_ID_RE = re.compile(r"^([0-9a-f]{16}):v(\d+):(\d{5})$")


def make_chunk_id(doc_id: str, version: int, seq: int) -> str:
    cid = f"{doc_id}:v{version}:{seq:05d}"
    if not CHUNK_ID_RE.fullmatch(cid):
        raise ValueError(f"非法 chunk_id: {cid}")
    return cid


def parse_chunk_id(cid: str) -> tuple[str, int, int]:
    if not (m := CHUNK_ID_RE.fullmatch(cid)):
        raise ValueError(f"非法 chunk_id: {cid}")
    return m.group(1), int(m.group(2)), int(m.group(3))


class Chunk(BaseModel):
    chunk_id: str = Field(pattern=CHUNK_ID_RE.pattern)  # make_chunk_id(doc_id, version, seq)，确定性生成，重切时可按 doc_id 整体删除
    doc_id: str = Field(pattern=DOC_ID_PATTERN)
    version: int
    seq: int                    # 文档内顺序
    type: ChunkType
    text: str                   # 原文（Markdown），给 LLM 和前端展示用
    embed_text: str             # 向量化用：标题路径 + 上下文前缀 + text
    content_hash: str           # text 归一化后的 sha256（§11），召回后去重用
    tokens: int                 # text 的 token 数（bge-m3 计数），上下文构造算预算用
    section_path: list[str]     # ["第2章 部署", "2.1 环境准备"]
    section_idx: int            # 所在节在文档内的编号（标题树先序遍历顺序），整节取回用
    section_tokens: int         # 所在节的总 token 数，决定能否整节放入上下文
    page_start: int | None
    page_end: int | None
    prev_id: str | None = None  # 同一文档中前后相邻的 text/code chunk
    next_id: str | None = None
    image_ids: list[str] = Field(default_factory=list)    # 本 chunk 关联的图片 chunk
    related_ids: list[str] = Field(default_factory=list)  # 代码与引导段落等互相关联的 chunk
    image_path: str | None = None  # 仅 image chunk
    searchable: bool = True     # 文档内重复、模板文本为 False（§11）：不参与检索，保留给上下文扩展
    meta: dict = Field(default_factory=dict)


class Draft(BaseModel):
    type: ChunkType
    text: str
    blocks: list[Block]         # 来源 Block：算页码、找图片占位符
    tokens: int
    section_path: list[str]
    section_idx: int
    section_tokens: int
    image_path: str | None = None
    meta: dict = Field(default_factory=dict)


class Section(BaseModel):
    heading: str | None          # 根节点为 None（对应文档标题之前的内容）
    level: int
    path: list[str]              # 从根到本节的标题列表
    idx: int                     # 先序遍历编号，根节点为 0
    body: list[Block]            # 本节标题之后、第一个子标题之前的 Block
    children: list["Section"]
    tokens: int = 0


class Unit(BaseModel):
    blocks: list[Block]
    kind: Literal["text", "code", "table", "list", "equation"]
    tokens: int
    hard_break: bool = False
