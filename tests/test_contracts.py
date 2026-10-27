"""契约模型：代码与文档逐字一致，以及跨模块的不变量（总揽.md §4.4）。"""

import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError
from pymilvus import DataType, MilvusClient

from ragagent.context.models import ContextBlock, ImageRef
from ragagent.graph.state import RAGState
from ragagent.index.models import ChunkFields, ChunkRow, EmbedItem, to_row
from ragagent.index.schema import create_collection
from ragagent.ingest.chunking.models import Chunk, make_chunk_id, parse_chunk_id
from ragagent.ingest.models import Block, CleanDocument
from ragagent.preprocess.schema import Preprocessed, Rewrite, Scope
from ragagent.prompt.models import ImagePart, Message, TextPart
from ragagent.retrieval.models import Expansion, Hit, RetrievalResult

DOCS = Path(__file__).resolve().parents[1] / "docs"
# docs/ 按 架构与关键信息 / 已完成的计划 / 计划 三类分目录，契约定义只出现在「架构与关键信息」
DOC_GLOB = "架构与关键信息/*.md"
FENCE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)

# 模块 -> 必须与文档逐字一致的定义
CONTRACTS = {
    "ragagent.ingest.models": ["BlockType", "DOC_ID_PATTERN", "Block", "CleanDocument"],
    "ragagent.ingest.chunking.models": ["ChunkType", "CHUNK_ID_RE", "make_chunk_id", "parse_chunk_id",
                                        "Chunk", "Draft", "Section", "Unit"],
    "ragagent.index.models": ["EmbedItem", "EmbedTextRequest", "EmbedImageRequest", "EmbedResponse",
                              "RerankRequest", "RerankPairsRequest", "RerankResponse",
                              "ChunkFields", "ChunkRow", "to_row"],
    "ragagent.index.schema": ["create_collection"],
    "ragagent.retrieval.models": ["Hit", "ExpansionRole", "Expansion", "RetrievalResult"],
    "ragagent.preprocess.schema": ["Scope", "Preprocessed", "Rewrite"],
    "ragagent.context.models": ["ImageRef", "ContextBlock"],
    "ragagent.prompt.models": ["TextPart", "ImagePart", "Part", "Message", "PromptPackage"],
    "ragagent.graph.state": ["RAGState"],
}
# 文档里的模块内部结构，由各模块自己实现，不属于 P0 契约
INTERNAL = {"ParseJob", "StreamResult", "Sentence"}


def _name(node: ast.stmt) -> str | None:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
        return node.name
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def _defs(tree: ast.Module) -> dict[str, str]:
    """顶层定义 -> 去掉注释和位置信息后的 AST dump。"""
    return {n: ast.dump(node) for node in tree.body if (n := _name(node))}


def _doc_defs() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for md in DOCS.glob(DOC_GLOB):
        for block in FENCE.findall(md.read_text(encoding="utf-8")):
            if block[:1] in " \t":      # 缩进的块是上一块的续写片段
                continue
            for name, dump in _defs(ast.parse(block)).items():
                out.setdefault(name, set()).add(dump)
    return out


DOC_DEFS = _doc_defs()


@pytest.mark.parametrize(
    ("module", "name"), [(m, n) for m, names in CONTRACTS.items() for n in names])
def test_matches_doc(module, name):
    code = _defs(ast.parse(inspect.getsource(importlib.import_module(module))))
    assert name in code, f"{module} 缺少 {name}"
    assert DOC_DEFS.get(name) == {code[name]}, f"{name} 与文档不一致（或文档里有多个版本）"


def test_every_doc_model_is_covered():
    covered = {n for names in CONTRACTS.values() for n in names}
    models = set()
    for md in DOCS.glob(DOC_GLOB):
        for block in FENCE.findall(md.read_text(encoding="utf-8")):
            if block[:1] in " \t":
                continue
            for node in ast.parse(block).body:
                if isinstance(node, ast.ClassDef) and {ast.unparse(b) for b in node.bases} & (
                        {"BaseModel", "TypedDict"} | covered):
                    models.add(node.name)
    assert models - covered - INTERNAL == set()


# ---- chunk_id / doc_id

DOC = "0123456789abcdef"


def test_chunk_id_roundtrip():
    cid = make_chunk_id(DOC, 3, 42)
    assert cid == f"{DOC}:v3:00042"
    assert parse_chunk_id(cid) == (DOC, 3, 42)


@pytest.mark.parametrize("args", [(DOC.upper(), 1, 0), (DOC[:15], 1, 0), (DOC, 1, 100000), (DOC, -1, 0)])
def test_make_chunk_id_rejects(args):
    with pytest.raises(ValueError):
        make_chunk_id(*args)


@pytest.mark.parametrize("cid", [f"{DOC}:v1:0001", f"{DOC}:1:00001", f"{DOC}:v1:00001\n", "x" * 16 + ":v1:00001"])
def test_parse_chunk_id_rejects(cid):
    with pytest.raises(ValueError):
        parse_chunk_id(cid)


def test_doc_id_pattern():
    with pytest.raises(ValidationError):
        CleanDocument(doc_id="../../etc", version=1, source_path="a.md", file_type="md",
                      parser="text", title="a", content_hash="h", blocks=[])


# ---- Chunk -> Milvus 行

def _chunk(**kw) -> Chunk:
    base = dict(chunk_id=make_chunk_id(DOC, 1, 0), doc_id=DOC, version=1, seq=0, type="text",
                text="正文", embed_text="第1章\n正文", content_hash="c" * 64, tokens=2,
                section_path=["第1章", "1.1"], section_idx=1, section_tokens=2,
                page_start=None, page_end=None)
    return Chunk(**base | kw)


def _doc() -> CleanDocument:
    return CleanDocument(doc_id=DOC, version=1, source_path="a/b.pdf", file_type="pdf",
                         parser="mineru-vlm", title="B", content_hash="h",
                         blocks=[Block(type="paragraph", text="正文")])


class _FakeClient:
    create_schema = MilvusClient.create_schema
    prepare_index_params = MilvusClient.prepare_index_params

    def create_collection(self, name, schema, index_params):
        self.name, self.schema, self.index_params = name, schema, index_params


def test_to_row_maps_none_and_doc_fields():
    row = to_row(_chunk(), _doc(), [0.0] * 4)
    assert row["page_start"] == row["page_end"] == -1
    assert row["prev_id"] == row["next_id"] == row["image_path"] == ""
    assert row["section_path"] == "第1章 > 1.1"
    assert (row["title"], row["source_path"], row["file_type"]) == ("B", "a/b.pdf", "pdf")
    assert row["searchable"] is True


def test_collection_fields_match_row():
    client = _FakeClient()
    create_collection(client, "chunks", dim=1024)
    fields = {f.name: f for f in client.schema.fields}
    assert set(fields) - {"sparse"} == set(ChunkRow.model_fields)
    assert fields["dense"].params["dim"] == 1024
    assert fields["chunk_id"].is_primary
    assert fields["embed_text"].params["enable_analyzer"]
    assert fields["sparse"].dtype == DataType.SPARSE_FLOAT_VECTOR
    assert [f.name for f in client.schema.functions] == ["bm25"]
    assert {i.field_name for i in client.index_params} >= {"dense", "sparse", "searchable"}


def test_chunk_fields_forbid_extra():
    row = to_row(_chunk(), _doc(), [])
    del row["dense"]
    ChunkFields(**row)
    with pytest.raises(ValidationError):
        ChunkFields(**row, skip_index=True)


def test_hit_and_expansion_from_row():
    row = to_row(_chunk(), _doc(), [])
    del row["dense"]
    hit = Hit(**row, score=0.5)
    assert Hit(**hit.model_dump()) == hit
    assert Expansion(**row, role="neighbor").role == "neighbor"
    with pytest.raises(ValidationError):
        Expansion(**row, role="sibling")
    assert RetrievalResult().model_dump() == {"top_score": 0.0, "degraded": False,
                                              "dense_only": False, "timings_ms": {}}


@pytest.mark.parametrize("path", ["", "/etc/passwd", "a/../../b.png"])
def test_embed_item_rejects_path(path):
    with pytest.raises(ValidationError):
        EmbedItem(image_path=path, text="")


# ---- 其他契约

def test_preprocess_caps():
    p = Preprocessed(standalone_question=" q ", extra_queries=[" ", "a" * 300, *"bcdef"],
                     scope=Scope(doc_titles=["t"] * 9))
    assert p.standalone_question == "q"
    assert p.extra_queries[0] == "a" * 200 and len(p.extra_queries) == 4
    assert len(p.scope.doc_titles) == 5
    assert Rewrite(queries=["  x ", ""]).queries == ["x"]


def test_message_discriminated_union():
    m = Message(role="user", parts=[TextPart(text="hi"),
                                    ImagePart(media_type="image/png", data_b64="AA==", chunk_id="c")])
    back = Message.model_validate(m.model_dump())
    assert [type(p) for p in back.parts] == [TextPart, ImagePart]


def test_context_block_images_default():
    b = ContextBlock(cite=1, doc_id=DOC, title="B", section_path="", page_start=-1, page_end=-1,
                     text="t", chunk_ids=["c"], hit_ids=["c"], score=0.1)
    assert b.images == []
    b.images.append(ImageRef(chunk_id="c", image_path="i.png", caption="", send_original=False))


def test_rag_state_fields():
    assert len(RAGState.__annotations__) == 22
    assert not issubclass(RAGState, BaseModel)
