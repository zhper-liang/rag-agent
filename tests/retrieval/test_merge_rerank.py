import asyncio
import copy
from types import SimpleNamespace

import httpx
import pytest

from ragagent.graph.deps import Deps
from ragagent.graph.nodes.routes import route_after_preprocess, route_after_rerank, route_after_rewrite
from ragagent.retrieval.merge import dedup_by_content, fetch_candidates, merge
from ragagent.retrieval.rerank import rerank, select
from tests.context.helpers import A, B, chunk, config
from tests.retrieval.fakes import FakeMilvus, row


def test_merge_takes_max_then_latest_version_before_limit():
    old = chunk(0, version=1, score=.99)
    new = chunk(0, version=2, score=.3)
    other = chunk(1, doc_id=B, score=.6)
    better = {**new, "score": .8}
    before = copy.deepcopy([old, new, other, better])
    merged = merge([[old, new], [other, better]], 2)
    assert [h["score"] for h in merged] == [.8, .6]
    assert all(h["version"] == 2 for h in merged if h["doc_id"] == A)
    assert [old, new, other, better] == before
    assert merge([[new]], 0) == []


def test_content_dedup_keeps_highest_scored_copy_after_fetch():
    first = chunk(0, text="same", score=.4)
    best = chunk(1, doc_id=B, text="same", score=.9)
    unique = chunk(2, text="other", score=.7)
    assert dedup_by_content([first, best, unique]) == [best, unique]
    fake = FakeMilvus([row(first), row(best), row(unique)])
    fetched = fetch_candidates(fake, "l2_fake", [first, best, unique], config().retrieval)
    assert [h.chunk_id for h in fetched] == [best["chunk_id"], unique["chunk_id"]]
    assert "searchable" not in fake.calls[0][1]["filter"]
    assert fetch_candidates(fake, "l2_fake", [], config().retrieval) == []


def test_select_threshold_inclusive_per_doc_cap_and_top_n_without_mutation():
    hits = [chunk(i) for i in range(5)] + [chunk(5, doc_id=B)]
    cfg = config(**{"rerank.per_doc_cap": 2, "rerank.top_n": 3}).rerank
    before = copy.deepcopy(hits)
    selected = select(hits, [.9, .8, .7, .3, .2, .3], cfg)
    assert [h["chunk_id"] for h in selected] == [hits[0]["chunk_id"], hits[1]["chunk_id"], hits[5]["chunk_id"]]
    assert selected[-1]["rerank_score"] == .3
    assert hits == before


@pytest.mark.parametrize("scores", [[.5], [float("nan"), .5], [1.1, .5], [-.1, .5]])
def test_invalid_rerank_score_response_rejected(scores):
    with pytest.raises(ValueError):
        select([chunk(0), chunk(1)], scores, config().rerank)


def test_rerank_real_http_stub_uses_standalone_question_and_embed_text(embed_url):
    async def run():
        async with httpx.AsyncClient(base_url=embed_url) as client:
            deps = Deps(config(), len, rerank_client=client)
            candidates = [chunk(0, text="not relevant", embed_text="Milvus DISKANN enableDisk"), chunk(1, embed_text="other")]
            hits, result = await rerank("Milvus DISKANN", candidates, deps, {"dense_only": True})
            assert hits[0].chunk_id == candidates[0]["chunk_id"]
            assert hits[0].rerank_score == result.top_score == 1
            assert result.dense_only and not result.degraded
            assert "rerank" in result.timings_ms
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["http", "timeout", "mismatch", "invalid"])
def test_rerank_failures_degrade_and_grade_directly_passes(mode):
    async def handler(request):
        if mode == "http":
            return httpx.Response(500)
        if mode == "timeout":
            raise httpx.ReadTimeout("offline")
        return httpx.Response(200, json={"scores": [.2] if mode == "mismatch" else [2., .5]})

    async def run():
        async with httpx.AsyncClient(base_url="http://stub", transport=httpx.MockTransport(handler)) as client:
            deps = Deps(config(), len, rerank_client=client)
            first, best = chunk(0, score=.2), chunk(1, score=.9)
            hits, result = await rerank("question", [first, best], deps)
            assert [h.chunk_id for h in hits] == [best["chunk_id"], first["chunk_id"]]
            assert result.degraded and all(h.rerank_score is None for h in hits)
            assert route_after_rerank({"retrieval": result.model_dump(), "retry_count": 0}, SimpleNamespace(context=deps)) == "build_context"
    asyncio.run(run())


def test_top_score_taken_before_threshold_filter_and_empty_skips_http():
    async def run():
        def handler(request):
            return httpx.Response(200, json={"scores": [.8]})
        async with httpx.AsyncClient(base_url="http://stub", transport=httpx.MockTransport(handler)) as client:
            deps = Deps(config(**{"rerank.threshold": .9}), len, rerank_client=client)
            hits, result = await rerank("question", [chunk()], deps)
            assert not hits and result.top_score == .8 and not result.degraded
            hits, result = await rerank("q", [], deps)
            assert hits == [] and result.top_score == 0
    asyncio.run(run())


# 分数相对 grade.pass_score 给出，校准改值后边界仍然被测到。
@pytest.mark.parametrize("delta,retries,degraded,expected", [
    (0, 0, False, "build_context"), (-.01, 0, False, "rewrite"),
    (-.01, 1, False, "rewrite"), (-.01, 2, False, "fallback"),
    (None, 2, True, "build_context"),
])
def test_grade_three_branches_and_degradation(delta, retries, degraded, expected):
    cfg = config()
    score = 0 if delta is None else cfg.grade.pass_score + delta
    runtime = SimpleNamespace(context=Deps(cfg, len))
    assert route_after_rerank({"retrieval": {"top_score": score, "degraded": degraded}, "retry_count": retries}, runtime) == expected


@pytest.mark.parametrize("intent,expected", [("rag", "retrieve"), ("chat", "generate"), ("clarify", "clarify")])
def test_intent_routes(intent, expected):
    assert route_after_preprocess({"intent": intent}) == expected


def test_rewrite_routes():
    assert route_after_rewrite({"queries": []}) == "fallback"
    assert route_after_rewrite({"queries": ["new"]}) == "retrieve"
