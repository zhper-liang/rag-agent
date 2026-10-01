import hashlib
from pathlib import Path

import yaml

from ragagent.config import Config
from ragagent.ingest.chunking.models import make_chunk_id

A, B = "0123456789abcdef", "aaaaaaaaaaaaaaaa"


def config(**updates):
    raw = yaml.safe_load(Path("config.yaml").read_text())
    for dotted, value in updates.items():
        keys = dotted.split(".")
        target = raw
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return Config.model_validate(raw)


def chunk(seq=0, *, doc_id=A, version=1, role=None, **updates):
    text = updates.get("text", f"正文 {seq}")
    c = dict(chunk_id=make_chunk_id(doc_id, version, seq), doc_id=doc_id, version=version, seq=seq,
             tokens=10, section_idx=1, section_tokens=40, type="text", text=text,
             embed_text=text, content_hash=hashlib.sha256(text.encode()).hexdigest(), title="手册",
             section_path="章 > 节", source_path="manual.md", file_type="md", page_start=-1,
             page_end=-1, image_path="", prev_id="", next_id="", image_ids=[], related_ids=[],
             searchable=True, meta={}, score=0.9, rerank_score=0.9)
    c.update(updates)
    if role:
        c.pop("score")
        c.pop("rerank_score")
        c["role"] = role
    return c


def link(*chunks):
    for left, right in zip(chunks, chunks[1:]):
        left["next_id"] = right["chunk_id"]
        right["prev_id"] = left["chunk_id"]
    return chunks
