import asyncio
from types import SimpleNamespace

import httpx
import pytest

from ragagent.graph.deps import Deps
from ragagent.ingest.chunking.models import make_chunk_id
from ragagent.retrieval.expand import expand, section_expr
from ragagent.retrieval.hybrid import LIGHT_FIELDS, RetrievalUnavailable, hybrid_many, hybrid_one, retrieve
from tests.context.helpers import A, B, chunk, config, link
from tests.retrieval.fakes import FakeMilvus, row


def test_hybrid_request_fields_weights_and_searchable_filter():
    fake = FakeMilvus([row(chunk())])
    result = hybrid_one(fake, "l2_fake", "question", [1.] * 1024, "searchable == true", config().retrieval)
    args = fake.calls[0][1]
    assert result[0]["score"] == 1
    assert args["output_fields"] == LIGHT_FIELDS
    assert args["reqs"][0].param["params"]["search_list"] == config().retrieval.search_list
    assert args["reqs"][1].data == ["question"]
    assert args["reqs"][0].expr == args["reqs"][1].expr == "searchable == true"


def test_multi_query_threads_concurrent_and_batch_embed_http(embed_url):
    async def run():
        fake = FakeMilvus([row(chunk())], delay=.05)
        async with httpx.AsyncClient(base_url=embed_url) as client:
            deps = Deps(config(), len, embed_client=client, milvus=fake)
            hits, result = await retrieve(["a", "b", "c"], {"doc_ids": [A]}, deps)
            assert len(hits) == 1 and fake.max_active >= 2
            assert not result.dense_only
            assert set(result.timings_ms) == {"embed", "hybrid", "fetch"}
            assert all('doc_id in ["0123456789abcdef"]' in args["reqs"][0].expr for kind, args in fake.calls if kind == "hybrid")
    asyncio.run(run())


def test_failed_query_is_ignored_if_another_succeeds():
    fake = FakeMilvus([row(chunk())], failures={"bad": "both"})
    results, dense_only = asyncio.run(hybrid_many(fake, "l2_fake", ["bad", "ok"], [[1.], [2.]], "searchable == true", config().retrieval))
    assert len(results) == 1 and not dense_only
    assert sum(kind == "dense" for kind, _ in fake.calls) == 1


def test_hybrid_failure_retries_once_dense_and_all_failure_503():
    fake = FakeMilvus([row(chunk())], failures={"bad": "hybrid"})
    results, dense_only = asyncio.run(hybrid_many(fake, "l2_fake", ["bad"], [[1.]], "searchable == true", config().retrieval))
    assert results and dense_only
    assert [kind for kind, _ in fake.calls] == ["hybrid", "dense"]
    fake.failures = {"bad": "both"}
    with pytest.raises(RetrievalUnavailable) as caught:
        asyncio.run(hybrid_many(fake, "l2_fake", ["bad"], [[1.]], "searchable == true", config().retrieval))
    assert caught.value.status_code == 503


def test_filter_validation_before_embed_or_milvus():
    with pytest.raises(ValueError):
        asyncio.run(retrieve(["q"], {"doc_ids": ['x" or true']}, Deps(config(), len)))


def test_embed_failure_wrong_shape_and_nonfinite_are_503():
    async def run(payload):
        def handler(request):
            return httpx.Response(200, json=payload)
        async with httpx.AsyncClient(base_url="http://stub", transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(RetrievalUnavailable, match="编码服务不可用"):
                await retrieve(["q"], None, Deps(config(), len, embed_client=client, milvus=FakeMilvus([])))
    for payload in ({"vectors": []}, {"vectors": [[1.]]}, {"bad": "response"}):
        asyncio.run(run(payload))


def test_empty_queries_skip_services():
    hits, result = asyncio.run(retrieve([], None, Deps(config(), len)))
    assert hits == [] and result.top_score == 0


def test_expand_batched_section_companions_and_searchable_false():
    text = chunk(1, tokens=10, section_tokens=50, meta={"split_part": "1/2"})
    sibling = chunk(2, meta={"split_part": "2/2"}, searchable=False)
    code = chunk(3, type="code", related_ids=[text["chunk_id"]])
    lead = chunk(0, searchable=False)
    image = chunk(4, type="image", image_path="clean/a.png", meta={"anchor_id": text["chunk_id"]})
    # text 自身命中不能再作为扩展返回；code 的 lead 单独取回。
    code["related_ids"] = [lead["chunk_id"]]
    text["image_ids"] = [image["chunk_id"]]
    link(lead, text, sibling, code)
    fake = FakeMilvus([row(c) for c in [lead, text, sibling, code, image]])
    expanded = expand(fake, "l2_fake", [text, code], config())
    types = {(e.chunk_id, e.role) for e in expanded[text["chunk_id"]]}
    assert (sibling["chunk_id"], "section") in types
    assert (sibling["chunk_id"], "split_sibling") in types
    assert (image["chunk_id"], "attachment") in types
    assert any(e.role == "lead_in" for e in expanded[code["chunk_id"]])
    assert all(e.chunk_id not in {text["chunk_id"], code["chunk_id"]} for values in expanded.values() for e in values)
    assert len(fake.calls) == 2
    assert all("searchable == true" not in args["filter"] for _, args in fake.calls)
    assert fake.calls[1][1]["limit"] == 200


def test_expand_large_section_only_neighbors_image_anchor_and_split_group_guard():
    hit = chunk(1, section_tokens=9999, meta={"split_part": "1/2"})
    other_group = chunk(2, meta={"split_part": "1/3"})
    link(hit, other_group)
    image = chunk(4, type="image", image_path="clean/img.png", meta={"anchor_id": other_group["chunk_id"]})
    fake = FakeMilvus([row(c) for c in [hit, other_group, image]])
    expanded = expand(fake, "l2_fake", [hit, image], config())
    assert {e.role for e in expanded[hit["chunk_id"]]} == {"neighbor"}
    assert {e.role for e in expanded[image["chunk_id"]]} == {"anchor"}
    assert len(fake.calls) == 1


def test_expand_images_per_chunk_and_global_and_version_guard():
    hits = [chunk(i) for i in range(3)]
    images = [chunk(10 + i, type="image", image_path="clean/a.png") for i in range(9)]
    for i, h in enumerate(hits):
        h["image_ids"] = [c["chunk_id"] for c in images[i * 3:i * 3 + 3]]
    fake = FakeMilvus([row(c) for c in hits + images])
    result = expand(fake, "l2_fake", hits, config())
    attachments = [e for es in result.values() for e in es if e.role == "attachment"]
    assert len({e.chunk_id for e in attachments}) == 4
    assert all(sum(e.role == "attachment" for e in es) <= 2 for es in result.values())
    foreign = chunk(20, doc_id=B)
    hits[0]["prev_id"] = foreign["chunk_id"]
    fake.rows.append(row(foreign))
    result = expand(fake, "l2_fake", hits, config())
    assert all(e.doc_id == A for values in result.values() for e in values)


@pytest.mark.parametrize("keys", [[('bad" or true', 1, 1)], [(A, '1 or true', 1)], [(A, True, 1)], [(A, 1, '2) or true')]])
def test_section_query_values_cannot_inject(keys):
    with pytest.raises(ValueError):
        section_expr(keys)


def test_expand_corrupt_id_rejected_before_query():
    hit = chunk(prev_id='bad" or true')
    fake = FakeMilvus([])
    with pytest.raises(ValueError):
        expand(fake, "l2_fake", [hit], config())
    assert fake.calls == []
