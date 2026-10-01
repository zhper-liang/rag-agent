import json
import math
from pathlib import Path

import pytest

from eval.align import align, align_gold
from eval.metrics import ndcg_at_k, percentile, recall_at_k, reciprocal_rank


def chunks():
    return [json.loads(s) for s in Path("tests/fixtures/chunks.jsonl").read_text().splitlines()]


def row(cid, text, overlap=0, seq=0, version=1):
    return dict(chunk_id=cid, text=text, meta={"overlap_chars": overlap}, seq=seq,
                version=version, doc_id="doc", type="text")


def test_align_fixture_single_cross_overlap_and_stale():
    cs = chunks()[:4]
    assert align("Milvus 的主配置文件是 milvus.yaml", cs) == {cs[1]["chunk_id"]}
    assert align("修改前先备份原文件，改完需要重启所有组件才会生效。使用 DISKANN", cs) == {cs[1]["chunk_id"], cs[2]["chunk_id"]}
    assert align("改完需要重启所有组件才会生效。", cs) == {cs[1]["chunk_id"]}
    assert align("改版后已删掉的片段", cs) is None


def test_exact_boundaries_and_overlap_not_double_counted():
    cs = [row("a", "abcdef"), row("b", "defghi", 3)]
    assert align("abc", cs) == {"a"}
    assert align("ghi", cs) == {"b"}
    assert align("efg", cs) == {"a", "b"}
    assert align("def", cs) == {"a"}
    assert align("", cs) is None
    assert align("x", []) is None
    with pytest.raises(ValueError):
        align("a", [row("bad", "a", 2)])


def test_gold_uses_latest_version_and_orders_seq():
    cs = [row("old", "obsolete", version=1), row("b", "world", seq=1, version=2), row("a", "hello", version=2)]
    assert align_gold([dict(doc_id="doc", version=1, quote="lowo")], cs) == [{"a", "b"}]
    assert align_gold([dict(doc_id="doc", version=1, quote="obsolete")], cs) is None


def test_hand_calculated_metrics_fragment_denominator():
    # 2 gold 片段；第1片段跨 a/a2，任意一个命中即可。分母2，非3 chunks。
    gold = [{"a", "a2"}, {"b"}]
    ranked = ["noise", "a2", "b", "a"]
    assert recall_at_k(ranked, gold, 2) == 1 / 2
    assert recall_at_k(ranked, gold, 3) == 1
    assert reciprocal_rank(ranked, gold) == 1 / 2
    # DCG = 1/log2(3)+1/log2(4)；IDCG = 1+1/log2(3)，替代a不再计分。
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranked, gold, 8) == pytest.approx(expected)
    assert ndcg_at_k(["a", "a2", "b"], gold, 8) == pytest.approx(1.5 / (1 + 1 / math.log2(3)))
    assert ndcg_at_k(["a", "b"], gold, 8) == 1
    assert recall_at_k(["x"], gold, 8) == 0
    assert reciprocal_rank(["x"], gold) == 0
    assert ndcg_at_k(["x"], gold, 8) == 0
    with pytest.raises(ValueError):
        recall_at_k([], [], 8)


def test_same_chunk_covers_multiple_quotes():
    assert ndcg_at_k(["a"], [{"a"}, {"a"}], 8) == 1
    assert percentile([1, 2, 3], .5) == 2
    assert percentile([1, 2, 3], .95) == pytest.approx(2.9)
