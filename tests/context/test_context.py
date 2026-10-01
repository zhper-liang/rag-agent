"""上下文构造 §11 的八条验证，逐条命名；附加边界与随机不变量。"""
import io
import json
import random
from pathlib import Path

import pytest
from PIL import Image

from ragagent.context.budget import context_usage, fill
from ragagent.context.build import build_context
from ragagent.context.images import prepare_image, resolve_image_path
from ragagent.context.merge import merge_blocks
from tests.context.helpers import A, B, chunk, config, link


def test_01_total_cost_with_chunks_images_and_overhead_is_within_budget():
    cfg = config(**{"context.budget_tokens": 95, "rerank.top_n": 2,
                    "context.block_overhead_tokens": 10, "context.token_safety_factor": 1.1})
    hits = [chunk(i, tokens=20) for i in range(4)]
    exp = {h["chunk_id"]: [chunk(i + 10, tokens=10, role="lead_in")] for i, h in enumerate(hits)}
    chosen, roles, attach, send = fill(hits, exp, cfg)
    usage = context_usage(chosen, roles, send, cfg)
    blocks = merge_blocks(chosen, roles, send)
    assert usage["total"] == sum([usage["P1"], usage["P2"], usage["P3"], usage["P4"],
                                   len(blocks) * cfg.context.block_overhead_tokens])
    assert usage["total"] <= cfg.context.budget_tokens
    assert len(chosen) == 3  # 伴随内容超过预扣段数时不能偷用段落开销。


def test_02_priority_retains_best_hits_then_companions_before_section():
    cfg = config(**{"context.budget_tokens": 60, "rerank.top_n": 1,
                    "context.block_overhead_tokens": 0, "context.token_safety_factor": 1})
    high, low, rejected = chunk(0, tokens=20), chunk(1, tokens=20, rerank_score=.8), chunk(2, tokens=30, rerank_score=.7)
    lead = chunk(3, tokens=20, role="lead_in")
    sec = chunk(4, tokens=20, role="section")
    unwanted = chunk(5, tokens=1, role="lead_in")
    chosen, roles, _, _ = fill([rejected, low, high], {high["chunk_id"]: [lead, sec], rejected["chunk_id"]: [unwanted]}, cfg)
    assert set(chosen) == {high["chunk_id"], low["chunk_id"], lead["chunk_id"]}
    assert roles[lead["chunk_id"]] == "lead_in"
    assert sec["chunk_id"] not in chosen and unwanted["chunk_id"] not in chosen


def test_03_chunk_once_and_shared_section_once():
    cfg = config()
    h1, h2, sec = chunk(0), chunk(2), chunk(1, role="section", searchable=False)
    exp = {h1["chunk_id"]: [sec, sec], h2["chunk_id"]: [sec]}
    chosen, roles, _, send = fill([h1, h2, h1], exp, cfg)
    blocks, _ = build_context([h1, h2, h1], exp, cfg)
    ids = [cid for b in blocks for cid in b.chunk_ids]
    assert len(ids) == len(set(ids)) == len(chosen) == 3
    assert len(blocks) == 1  # 整节即使没有 prev/next 链也是单引用。
    assert chosen[sec["chunk_id"]]["searchable"] is False
    assert context_usage(chosen, roles, send, cfg)["P3"] == 11


def test_04_split_merge_reconstructs_original_without_duplicate_overlap():
    left = chunk(0, text="ABCDEFGHI", meta={"split_part": "1/3", "overlap_chars": 0})
    mid = chunk(1, role="split_sibling", text="GHIJKLM", meta={"split_part": "2/3", "overlap_chars": 3})
    right = chunk(2, role="split_sibling", text="LMNOP", meta={"split_part": "3/3", "overlap_chars": 2})
    link(left, mid, right)
    blocks, _ = build_context([left], {left["chunk_id"]: [mid, right]}, config())
    assert blocks[0].text == "ABCDEFGHIJKLMNOP"
    assert len(blocks) == 1
    # 缺失前段时不能剥离首段的重叠：保留全文。
    blocks, _ = build_context([chunk(1, text="GHIJKLM", meta=mid["meta"])], {}, config())
    assert blocks[0].text == "GHIJKLM"


def test_05_section_path_changes_insert_markdown_heading():
    first, second = link(chunk(0, section_path="章 > 第一节"), chunk(1, section_path="章 > 第二节"))
    blocks, _ = build_context([first, second], {}, config())
    assert "\n\n## 第二节\n\n" in blocks[0].text
    assert blocks[0].section_path == "章 > 第一节"


def test_06_documents_by_best_score_chunks_by_sequence_and_degraded_fusion():
    first, second = link(chunk(0, rerank_score=.7), chunk(1, rerank_score=.6))
    other = chunk(2, doc_id=B, rerank_score=.9)
    blocks, _ = build_context([second, first, other], {}, config())
    assert [b.doc_id for b in blocks] == [B, A]
    assert blocks[1].chunk_ids == [first["chunk_id"], second["chunk_id"]]
    assert [b.cite for b in blocks] == [1, 2]
    first["rerank_score"], other["rerank_score"] = None, None
    first["score"], other["score"] = .99, .5
    blocks, _ = build_context([first, other], {}, config())
    assert blocks[0].doc_id == A and blocks[0].score == .99


def test_07_retained_hit_ids_cover_all_budget_retained_hits():
    cfg = config(**{"context.budget_tokens": 30, "context.block_overhead_tokens": 0,
                    "context.token_safety_factor": 1})
    hits = [chunk(0, tokens=20), chunk(1, tokens=20, rerank_score=.8)]
    companion = chunk(2, tokens=5, role="anchor")
    blocks, _ = build_context(hits, {hits[0]["chunk_id"]: [companion]}, cfg)
    ids = [cid for b in blocks for cid in b.hit_ids]
    assert ids == [hits[0]["chunk_id"]]
    assert companion["chunk_id"] in [cid for b in blocks for cid in b.chunk_ids]
    assert companion["chunk_id"] not in ids


def test_08_image_original_and_attachment_limits_and_path_rejection(tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    Image.new("RGBA", (2048, 1024), "blue").save(clean / "ok.png")
    cfg = config(**{"paths.clean": str(clean), "context.budget_tokens": 10000})
    imgs = [chunk(i, type="image", image_path="clean/ok.png") for i in range(3)]
    texts = [chunk(i + 10) for i in range(3)]
    exp = {t["chunk_id"]: [chunk(i + 30, type="image", role="attachment", image_path="clean/ok.png")
                           for i in range(j * 3, j * 3 + 3)] for j, t in enumerate(texts)}
    blocks, attachments = build_context(imgs + texts, exp, cfg)
    refs = [r for b in blocks for r in b.images]
    assert sum(r.send_original for r in refs) == 2
    assert len(attachments) == 4
    assert len({r.chunk_id for r in attachments}) == 4
    blob = prepare_image("clean/ok.png", cfg)
    with Image.open(io.BytesIO(blob)) as image:
        assert image.mode == "RGB" and image.format == "JPEG"
        assert image.size == (1024, 512)
    for path in ("/etc/passwd", "clean/../outside.jpg", "../clean/a.jpg", "raw/a.jpg", "C:\\secret.jpg", ""):
        with pytest.raises(ValueError):
            build_context([chunk(type="image", image_path=path)], {}, cfg)
    (clean / "escape.png").symlink_to(tmp_path / "outside.png")
    with pytest.raises(ValueError):
        resolve_image_path("clean/escape.png", clean)


def test_section_is_atomic_then_falls_back_to_immediate_neighbors():
    cfg = config(**{"context.budget_tokens": 45, "context.block_overhead_tokens": 0,
                    "context.token_safety_factor": 1})
    hit = chunk(2, tokens=20)
    prev, following, far = [chunk(i, tokens=20, role="section") for i in (1, 3, 4)]
    link(prev, hit, following, far)
    chosen, roles, _, _ = fill([hit], {hit["chunk_id"]: [prev, following, far]}, cfg)
    assert set(chosen) == {prev["chunk_id"], hit["chunk_id"]}
    assert roles[prev["chunk_id"]] == "neighbor"


def test_image_budget_missing_and_unreadable_fall_back_to_caption(tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    Image.new("RGB", (8, 8)).save(clean / "ok.png")
    (clean / "bad.png").write_text("not image")
    cfg = config(**{"paths.clean": str(clean), "context.budget_tokens": 1000})
    for path in ("clean/ok.png", "clean/missing.png", "clean/bad.png"):
        blocks, _ = build_context([chunk(type="image", image_path=path)], {}, cfg)
        assert not blocks[0].images[0].send_original
        assert "[图:" in blocks[0].text
    assert prepare_image("clean/bad.png", cfg) is None


def test_hit_image_attaches_to_anchor_not_own_block():
    anchor = chunk(0, role="anchor", text="锚点", page_start=3, page_end=3)
    image = chunk(1, type="image", image_path="clean/missing.png", meta={"anchor_id": anchor["chunk_id"]}, page_start=4, page_end=4)
    blocks, _ = build_context([image], {image["chunk_id"]: [anchor]}, config())
    assert len(blocks) == 1
    assert blocks[0].hit_ids == [image["chunk_id"]]
    assert blocks[0].page_start == 3 and blocks[0].page_end == 4
    assert "锚点" in blocks[0].text


def test_empty_and_oversized_hit_do_not_add_companions():
    assert build_context([], {}, config()) == ([], [])
    hit = chunk(tokens=99999)
    assert build_context([hit], {hit["chunk_id"]: [chunk(1, role="lead_in")]}, config()) == ([], [])


def test_randomized_budget_uniqueness_and_hit_coverage():
    rng = random.Random(42)
    for _ in range(100):
        cfg = config(**{"context.budget_tokens": rng.randint(200, 1500), "rerank.top_n": 8})
        hits = [chunk(i, tokens=rng.randint(1, 300), rerank_score=rng.random()) for i in range(8)]
        exp = {h["chunk_id"]: [chunk(j, tokens=rng.randint(1, 150), role="lead_in")
                               for j in (10 + i, 20 + i)] for i, h in enumerate(hits)}
        chosen, roles, _, send = fill(hits, exp, cfg)
        assert context_usage(chosen, roles, send, cfg)["total"] <= cfg.context.budget_tokens
        blocks, _ = build_context(hits, exp, cfg)
        ids = [cid for b in blocks for cid in b.chunk_ids]
        assert len(ids) == len(set(ids)) == len(chosen)
        assert {cid for b in blocks for cid in b.hit_ids} == {cid for cid in chosen if roles[cid] == "hit"}


def test_contract_hit_fixture_is_consumed_without_mutation():
    hits = json.loads(Path("tests/fixtures/hits.json").read_text())
    before = json.dumps(hits, sort_keys=True)
    blocks, _ = build_context(hits, {}, config())
    assert blocks and json.dumps(hits, sort_keys=True) == before
