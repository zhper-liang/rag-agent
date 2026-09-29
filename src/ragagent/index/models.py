"""向量服务的请求/响应和 Milvus 行结构（向量化建库.md §4.2、§5.1）。"""

from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, field_validator

from ragagent.ingest.chunking.models import Chunk
from ragagent.ingest.models import CleanDocument


class EmbedItem(BaseModel):
    image_path: str
    text: str

    @field_validator("image_path")
    @classmethod
    def _relative(cls, v: str) -> str:
        p = PurePosixPath(v)
        if not v or p.is_absolute() or ".." in p.parts:
            raise ValueError(f"image_path 必须是相对路径且不含 ..: {v!r}")
        return v


class EmbedTextRequest(BaseModel):
    texts: list[str]


class EmbedImageRequest(BaseModel):
    items: list[EmbedItem]


class EmbedResponse(BaseModel):
    vectors: list[list[float]]


class RerankRequest(BaseModel):
    query: str
    passages: list[str]
    max_length: int             # 取 rerank.max_length；query + passage 超出部分截断


class RerankPairsRequest(BaseModel):
    pairs: list[tuple[str, str]]
    max_length: int


class RerankResponse(BaseModel):
    scores: list[float]         # 经过 sigmoid，0~1，顺序与输入一致


class ChunkFields(BaseModel):
    model_config = ConfigDict(extra="forbid")   # 字段名拼错直接报错，不静默丢弃

    chunk_id: str
    doc_id: str
    version: int
    seq: int
    tokens: int
    section_idx: int
    section_tokens: int
    type: str
    text: str
    embed_text: str
    content_hash: str
    title: str
    section_path: str           # " > " 连接
    source_path: str
    file_type: str
    page_start: int             # 没有页码时 -1
    page_end: int
    image_path: str             # 非 image chunk 为 ""
    prev_id: str                # 没有时为 ""
    next_id: str
    image_ids: list[str]
    related_ids: list[str]
    searchable: bool
    meta: dict


class ChunkRow(ChunkFields):
    """写入 Milvus 的一行；sparse 由 BM25 Function 生成，不在这里。"""
    dense: list[float]


def to_row(c: Chunk, doc: CleanDocument, dense: list[float]) -> dict:
    none_to = {"page_start": -1, "page_end": -1, "image_path": "", "prev_id": "", "next_id": ""}
    base = c.model_dump(exclude={"section_path", *none_to})
    fixed = {k: d if getattr(c, k) is None else getattr(c, k) for k, d in none_to.items()}
    return ChunkRow(**base, **fixed, dense=dense, title=doc.title,
                    source_path=doc.source_path, file_type=doc.file_type,
                    section_path=" > ".join(c.section_path)).model_dump()
