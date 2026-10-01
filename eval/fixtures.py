"""显式 --fixture 模式，只消费 P0 夹具，不访问外部服务或构造真实基线。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def fixture_chunks():
    return [json.loads(line) for line in (ROOT / "chunks.jsonl").read_text().splitlines()]


def fixture_dataset():
    chunks = fixture_chunks()
    doc_id = chunks[0]["doc_id"]
    gold = [{"doc_id": doc_id, "version": 1, "quote": "使用 DISKANN 索引时，必须把 queryNode.enableDisk 设为 true"}]
    base = {"id": "fixture-single", "type": "single", "tags": ["keyword"],
            "turns": [{"question": "磁盘索引加载失败时先检查哪个开关？"}],
            "expect": {"intent": "rag", "standalone": None, "scope_doc_ids": [], "answerable": True},
            "gold": gold, "reference": "检查 queryNode.enableDisk 是否设为 true。",
            "gen": {"model": "fixture", "prompt_version": "eval-v1", "created": "fixture"}}
    return [base, {**base, "id": "fixture-multi", "type": "multi_chunk", "tags": [],
                   "gold": gold + [{"doc_id": doc_id, "version": 1, "quote": "Milvus 的主配置文件是 milvus.yaml"}]},
            {**base, "id": "fixture-multi-turn", "type": "multi_turn", "tags": [],
             "turns": [{"question": "用什么磁盘索引？", "answer": "DISKANN。"}, {"question": "它需要哪个开关？"}],
             "expect": {**base["expect"], "standalone": "DISKANN 需要哪个开关？"}},
            {**base, "id": "fixture-no-answer", "type": "unanswerable", "tags": [], "gold": [], "reference": "",
             "turns": [{"question": "下一季度售价是多少？"}], "expect": {**base["expect"], "answerable": False}},
            {**base, "id": "fixture-chat", "type": "chat", "tags": [], "gold": [], "reference": "",
             "expect": {**base["expect"], "intent": "chat", "answerable": False}},
            {**base, "id": "fixture-clarify", "type": "clarify", "tags": [], "gold": [], "reference": "",
             "expect": {**base["expect"], "intent": "clarify", "answerable": False}},
            {**base, "id": "fixture-scoped", "type": "scoped", "tags": [],
             "expect": {**base["expect"], "scope_doc_ids": [doc_id]}},
            {**base, "id": "fixture-stale", "gold": [{"doc_id": doc_id, "version": 1, "quote": "已被删除的段落"}]}]


class FixtureBackend:
    def __init__(self, regression=False):
        self.regression = regression
        self.hits = json.loads((ROOT / "hits.json").read_text())
        self.blocks = json.loads((ROOT / "context_blocks.json").read_text())

    async def __call__(self, case, tier):
        expect = case["expect"]
        rag = expect["intent"] == "rag"
        hits = [] if self.regression or not case["gold"] else self.hits
        return {"intent": expect["intent"], "standalone_question": expect.get("standalone") or case["turns"][-1]["question"],
                "candidates": hits, "hits": hits, "context_blocks": [] if self.regression else self.blocks,
                "soft_filters": {"doc_ids": expect.get("scope_doc_ids", [])},
                "retrieval": {"total_ms": 300 if self.regression else 100, "top_score": .93 if hits else .0},
                "route": "fallback" if rag and not expect["answerable"] else expect["intent"],
                "unanswerable_handled": not self.regression,
                "answer": (ROOT / "answers" / "basic.md").read_text() if tier == "full" else "",
                "gen_meta": {"post": {"citation_support_rate": .5 if self.regression else .95,
                                       "citation_validity_rate": 1.0, "citation_coverage": 1.0}},
                "ragas": {"faithfulness": .5 if self.regression else .95, "answer_relevancy": .9,
                          "context_precision": .9, "context_recall": .9, "answer_correctness": .9}}
