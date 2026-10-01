"""夹具与契约测试（开发计划.md §3.3）：改字段、改约定就红。"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pytest
import yaml

from ragagent.context.models import ContextBlock
from ragagent.ingest.chunking.models import Chunk, parse_chunk_id
from ragagent.ingest.models import Block
from ragagent.preprocess.schema import Preprocessed, Rewrite
from ragagent.prompt.env import RESERVED
from ragagent.retrieval.models import Hit

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
CITE = re.compile(r"\[(\d+)\]")
CODE_FENCE = re.compile(r"^`{3,}.*?^`{3,}", re.S | re.M)


def _json(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def _roundtrip(model, data):
    """model_dump 必须原样还原：字段改名、加字段、改默认值都会失败。"""
    assert model.model_validate(data).model_dump() == data


BLOCKS = _json("blocks.json")
CHUNKS = [json.loads(line) for line in (FIX / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
HITS = _json("hits.json")
CTX = _json("context_blocks.json")
BY_ID = {c["chunk_id"]: c for c in CHUNKS}
ANSWERS = {p.stem: p.read_text(encoding="utf-8") for p in sorted((FIX / "answers").glob("*.md"))}


def _content_hash(text: str) -> str:
    # 文档切分.md §11：NFKC、去掉所有空白后 sha256
    return hashlib.sha256(re.sub(r"\s", "", unicodedata.normalize("NFKC", text)).encode()).hexdigest()


# ---- blocks.json

def test_blocks():
    assert len(BLOCKS) == 5
    for b in BLOCKS:
        _roundtrip(Block, b)
    assert {b["type"] for b in BLOCKS} == {"heading", "paragraph", "table", "code", "image"}


# ---- chunks.jsonl

def test_chunks_schema():
    assert len(CHUNKS) == 5
    for c in CHUNKS:
        _roundtrip(Chunk, c)
        assert parse_chunk_id(c["chunk_id"]) == (c["doc_id"], c["version"], c["seq"])
        assert c["content_hash"] == _content_hash(c["text"])
        assert c["tokens"] <= CFG["chunking"]["max_tokens"]
        # embed_text = 标题路径 + 前缀 + 正文；整节 chunk 去掉与路径重复的首行标题
        assert c["embed_text"].split("\n", 1)[0].endswith(" > ".join(c["section_path"]))
        assert c["text"].split("\n", 2)[-1] in c["embed_text"]
    assert {c["type"] for c in CHUNKS} == {"text", "code", "image"}
    assert {c["meta"]["strategy"] for c in CHUNKS} >= {"section", "split", "code", "image"}


def test_chunks_prev_next_chain():
    chain = [c for c in CHUNKS if c["type"] != "image"]      # 图片不进链条
    for a, b in zip(chain, chain[1:]):
        assert (a["next_id"], b["prev_id"]) == (b["chunk_id"], a["chunk_id"])
    assert chain[0]["prev_id"] is None and chain[-1]["next_id"] is None
    assert all(c["prev_id"] is c["next_id"] is None for c in CHUNKS if c["type"] == "image")


def test_chunks_image_anchor_bidirectional():
    for img in (c for c in CHUNKS if c["type"] == "image"):
        anchor = BY_ID[img["meta"]["anchor_id"]]
        assert img["chunk_id"] in anchor["image_ids"]
        assert anchor["chunk_id"] in img["related_ids"]
        assert img["image_path"] and img["section_path"] == anchor["section_path"]


def test_chunks_code_lead_in():
    code = next(c for c in CHUNKS if c["type"] == "code")
    lead = BY_ID[code["related_ids"][0]]
    assert code["chunk_id"] in lead["related_ids"]
    assert lead["text"].endswith(code["meta"]["lead_in"])
    assert code["meta"]["lead_in"] in code["embed_text"] and code["meta"]["lead_in"] not in code["text"]


def test_chunks_split_overlap():
    parts = [c for c in CHUNKS if c["meta"]["strategy"] == "split"]
    assert [p["meta"]["split_part"] for p in parts] == ["1/2", "2/2"]
    first, second = parts
    ov = second["meta"]["overlap_chars"]
    assert ov > 0 and first["text"].endswith(second["text"][:ov])


# ---- hits.json

def test_hits():
    assert len(HITS) == 10
    for h in HITS:
        _roundtrip(Hit, h)
    scores = [h["rerank_score"] for h in HITS]
    assert scores == sorted(scores, reverse=True)
    ids = {h["chunk_id"] for h in HITS}
    assert any(h["next_id"] in ids for h in HITS)                # 至少一组前后相邻
    by_hash: dict[str, set[str]] = {}
    for h in HITS:
        by_hash.setdefault(h["content_hash"], set()).add(h["doc_id"])
    assert any(len(docs) > 1 for docs in by_hash.values())       # 跨文档内容重复


def test_hits_agree_with_chunks():
    for h in HITS:
        if c := BY_ID.get(h["chunk_id"]):
            assert h["section_path"] == " > ".join(c["section_path"])
            assert (h["text"], h["content_hash"], h["meta"]) == (c["text"], c["content_hash"], c["meta"])
            assert h["prev_id"] == (c["prev_id"] or "") and h["page_start"] == (c["page_start"] or -1)


# ---- context_blocks.json

def test_context_blocks():
    assert len(CTX) == 3
    for b in CTX:
        _roundtrip(ContextBlock, b)
    assert [b["cite"] for b in CTX] == [1, 2, 3]
    hits = {h["chunk_id"]: h for h in HITS}
    for b in CTX:
        assert set(b["hit_ids"]) <= set(b["chunk_ids"]) and set(b["hit_ids"]) <= set(hits)
        assert b["score"] == max(hits[i]["rerank_score"] for i in b["hit_ids"])
        assert {parse_chunk_id(i)[0] for i in b["chunk_ids"]} == {b["doc_id"]}


def test_context_block_merge_strips_overlap():
    b = CTX[0]
    first, second = (BY_ID[i] for i in b["chunk_ids"][:2])
    merged = first["text"] + second["text"][second["meta"]["overlap_chars"]:]
    assert b["text"].startswith(merged)
    assert [img["chunk_id"] for img in b["images"]] == second["image_ids"]


# ---- 预处理、改写

def test_preprocess_out():
    samples = _json("preprocess_out.json")
    for s in samples:
        _roundtrip(Preprocessed, s)
    assert {s["intent"] for s in samples} == {"rag", "chat", "clarify"}
    assert all((s["clarify_question"] is not None) == (s["intent"] == "clarify") for s in samples)


def test_rewrite_out():
    _roundtrip(Rewrite, _json("rewrite_out.json"))


# ---- answers/

def test_answer_citations():
    assert len(ANSWERS) == 3
    for text in ANSWERS.values():
        cites = {int(n) for n in CITE.findall(text)}
        assert cites and cites <= {b["cite"] for b in CTX}
        assert not any(CITE.search(block) for block in CODE_FENCE.findall(text))   # 代码块内不加引用


def test_answer_shapes():
    heading = CFG["prompt"]["supplement_heading"]
    assert CODE_FENCE.search(ANSWERS["code"])
    head, sep, tail = ANSWERS["supplement"].partition(f"\n{heading}\n")
    assert sep and CITE.search(head) and not CITE.search(tail)   # 补充说明之后不带引用
    assert "{{ supplement_heading }}" in (ROOT / "prompts" / "v1" / "system_rag.j2").read_text(encoding="utf-8")


# ---- Prompt 标签

def test_reserved_tags_cover_templates():
    used = set()
    for tpl in (ROOT / "prompts" / "v1").glob("*.j2"):
        used |= {m.lower() for m in re.findall(r"</?\s*([A-Za-z_]+)", tpl.read_text(encoding="utf-8"))}
    assert used and used <= set(RESERVED.split("|"))
