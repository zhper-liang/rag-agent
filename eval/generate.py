"""文档分层抽样 → structured_call → 逐字quote/措辞/向量去重过滤。"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
from collections import Counter, defaultdict
from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragagent.llm.observability import metrics_scope
from eval.calls import counted_calls, current_summary
from ragagent.config import load_config
from ragagent.llm.structured import structured_call

TYPES = {"single", "multi_chunk", "unanswerable", "multi_turn", "chat", "clarify", "scoped"}


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)
    answer: str | None = None


class Expect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Literal["rag", "chat", "clarify"]
    standalone: str | None = None
    scope_doc_ids: list[str] = Field(default_factory=list)
    answerable: bool


class Gold(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    version: int = Field(ge=1)
    quote: str = Field(min_length=1)


class GeneratedQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tags: list[Literal["keyword", "image"]] = Field(default_factory=list)
    turns: list[Turn] = Field(min_length=1)
    expect: Expect
    gold: list[Gold]
    reference: str


def bigram_jaccard(a: str, b: str) -> float:
    grams = lambda x: {x[i:i + 2] for i in range(len(x) - 1)}
    aa, bb = grams(a), grams(b)
    return len(aa & bb) / len(aa | bb) if aa or bb else float(a == b)


def cosine(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("embedding dimensions do not match")
    if any(not math.isfinite(v) for v in (*a, *b)):
        raise ValueError("embedding contains non-finite value")
    denom = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
    return sum(x * y for x, y in zip(a, b)) / denom if denom else 0.0


def sample_chunks(chunks, settings, *, seed=0):
    """确定性文档分层round-robin；跳过过短、不可检索及旧版本。"""
    latest = {}
    for c in chunks:
        latest[c["doc_id"]] = max(latest.get(c["doc_id"], 0), c["version"])
    docs = defaultdict(list)
    for c in chunks:
        if c.get("searchable", True) and c["tokens"] >= settings.min_chunk_tokens and c["version"] == latest[c["doc_id"]]:
            docs[c["doc_id"]].append(c)
    rng = random.Random(seed)
    keys = sorted(docs)
    rng.shuffle(keys)
    for rows in docs.values():
        rows.sort(key=lambda c: c["seq"])
        rng.shuffle(rows)
    result = []
    for i in range(max((len(rows) for rows in docs.values()), default=0)):
        for key in keys:
            if i < len(docs[key]):
                result.append(docs[key][i])
    return result


def validate_question(q: GeneratedQuestion, selected: list[dict], kind: str, settings) -> str | None:
    if kind not in TYPES:
        return "unknown_type"
    expected_intent = kind if kind in {"chat", "clarify"} else "rag"
    answerable = kind not in {"chat", "clarify", "unanswerable"}
    if q.expect.intent != expected_intent or q.expect.answerable != answerable:
        return "expect_mismatch"
    if answerable and (not q.gold or not q.reference.strip()):
        return "missing_gold"
    if not answerable and q.gold:
        return "unexpected_gold"
    if kind == "multi_turn" and (len(q.turns) < 2 or not q.expect.standalone or any(not t.answer for t in q.turns[:-1])):
        return "invalid_multi_turn"
    if kind != "multi_turn" and len(q.turns) != 1:
        return "invalid_turns"
    if kind == "multi_chunk" and (len(q.gold) < 2 or len({(g.doc_id, g.quote) for g in q.gold}) < 2):
        return "missing_multi_gold"
    doc_ids = {c["doc_id"] for c in selected}
    if kind == "scoped" and (not q.expect.scope_doc_ids or not set(q.expect.scope_doc_ids) <= doc_ids):
        return "invalid_scope"
    if kind != "scoped" and q.expect.scope_doc_ids:
        return "unexpected_scope"
    questions = [t.question for t in q.turns]
    for g in q.gold:
        if len(g.quote) < settings.min_quote_chars:
            return "short_quote"
        source = [c for c in selected if c["doc_id"] == g.doc_id and c["version"] == g.version and g.quote in c["text"]]
        if not source:
            return "quote_not_in_chunk"
        if any(bigram_jaccard(question, g.quote) > settings.max_overlap for question in questions):
            return "copied_question"
    if kind == "single" and any(c.get("type") == "image" for c in selected) and "image" not in q.tags:
        return "missing_image_tag"
    if kind == "multi_chunk":
        covered = {c["chunk_id"] for c in selected for g in q.gold
                   if c["doc_id"] == g.doc_id and c["version"] == g.version and g.quote in c["text"]}
        if len(covered) < 2:
            return "missing_multi_chunks"
    return None


async def generate(chunks, cfg, *, call=structured_call, embed, retrieve, seed=0,
                   on_accept=None, concurrency=1, resume_records=None):
    """预算外由CLI限制；文档名额预留、去重和落盘在同一事件循环中串行确认。"""
    settings = cfg.eval.generate
    if settings.max_regen < 0 or not 0 <= settings.max_overlap <= 1 or not 0 <= settings.dedup_similarity <= 1:
        raise ValueError("invalid eval.generate thresholds")
    if set(settings.counts) - TYPES or any(n < 0 for n in settings.counts.values()) or concurrency < 1:
        raise ValueError("invalid eval.generate counts/concurrency")
    sources = sample_chunks(chunks, settings, seed=seed)
    accepted, vectors = list(resume_records or []), []
    per_doc, reserved, rejects, missing = Counter(), Counter(), Counter(), Counter()
    for record in accepted:
        docs = record.get('gen', {}).get('source_doc_ids') or {g['doc_id'] for g in record['gold']}
        if not docs:
            raise ValueError('resuming answerless records requires source_doc_ids provenance')
        per_doc.update(set(docs))
        text = "\n".join(t['question'] for t in record['turns'])
        embedded = await embed([text]); cosine(embedded[0], embedded[0])
        vectors.append(embedded[0])
    initial = Counter(r['type'] for r in accepted)
    if any(initial[k] > settings.counts.get(k, 0) for k in initial) or any(n > settings.max_per_doc for n in per_doc.values()):
        raise ValueError('resume dataset exceeds configured count/doc limits')
    changed, sem = asyncio.Condition(), asyncio.Semaphore(concurrency)

    async def reserve(kind):
        nonlocal sources
        async with changed:
            while True:
                source = next((c for c in sources if per_doc[c['doc_id']]+reserved[c['doc_id']] < settings.max_per_doc), None)
                if source is None:
                    if any(reserved.values()):
                        await changed.wait()
                        continue
                    return None
                selected = [source]
                if kind == 'multi_chunk':
                    partners = [c for c in sources if c['chunk_id'] != source['chunk_id']
                        and per_doc[c['doc_id']]+reserved[c['doc_id']] < settings.max_per_doc]
                    partners.sort(key=lambda c: (c['doc_id'] != source['doc_id'], c.get('section_idx') != source.get('section_idx')))
                    if not partners:
                        return None
                    selected.append(partners[0])
                reserved.update({c['doc_id'] for c in selected})
                sources = sources[1:] + sources[:1]
                return selected

    async def one(kind):
        async with sem:
            selected = await reserve(kind)
            if not selected:
                missing[kind] += 1
                return
            success = False
            try:
                for attempt in range(settings.max_regen + 1):
                    messages = [SystemMessage(content=(
                        "生成中文RAG评测题，必须是用户口语且不能照抄原句。返回所要求的结构化schema。"
                        "gold只能是输入chunk中逐字摘录且长度达到要求的quote，不要用chunk_id。"
                        "single/scoped/multi_turn有答案；multi_chunk须结合两段，各给一条gold。"
                        "unanswerable在资料中确实无答案；chat闲聊；clarify缺关键信息须反问。"
                        "多轮前轮须有answer，末轮不填answer并给expect.standalone；gold和reference只回答末轮问题。"
                        "不要提问未在资料中解释的原因；资料仅给规则时问规则而非为什么。"
                        "single题必须独立清楚，不能用‘这段代码’或‘那个方案’代指未向用户展示的片段。"
                        "无答案题针对整套资料缺失的信息，不能只因当前片段没提到就判无答案。"
                        "scoped给scope_doc_ids。"
                        "精确词题标keyword；答案来自图片时标image。")),
                        HumanMessage(content=json.dumps({"type": kind, "min_quote_chars": settings.min_quote_chars,
                            "attempt": attempt, "chunks": [{k: c.get(k) for k in ("doc_id", "version", "text", "type", "title")} for c in selected]}, ensure_ascii=False))]
                    parsed, endpoint = await call(cfg.llm.scenes.eval, cfg.llm.endpoints, GeneratedQuestion, messages)
                    if parsed is None:
                        rejects['llm_failed'] += 1
                        continue
                    try:
                        q = GeneratedQuestion.model_validate(parsed)
                    except ValidationError:
                        rejects['invalid_schema'] += 1
                        continue
                    reason = validate_question(q, selected, kind, settings)
                    text = "\n".join(t.question for t in q.turns)
                    vector = None
                    if reason is None:
                        embedded = await embed([text])
                        if len(embedded) != 1:
                            raise ValueError('embedding service returned wrong count')
                        vector = embedded[0]; cosine(vector, vector)
                    if reason is None and kind == 'unanswerable':
                        found = await retrieve(q.turns[-1].question)
                        top = found.get('retrieval', {}).get('top_score', found.get('top_score'))
                        if top is None or not isinstance(top, (int, float)) or not math.isfinite(top):
                            raise ValueError('unanswerable filter requires actual rerank top_score')
                        if top >= settings.unanswerable_max_score:
                            reason = 'has_answer'
                    # 这里直到append/on_accept间没有await；并发的题不会越过去重检查。
                    if reason is None and (any(text == "\n".join(t['question'] for t in r['turns']) for r in accepted)
                                           or any(cosine(vector, v) > settings.dedup_similarity for v in vectors)):
                        reason = 'duplicate'
                    if reason:
                        rejects[reason] += 1
                        continue
                    record = q.model_dump(exclude_none=True)
                    record.update(id=f'q{len(accepted)+1:05d}', type=kind,
                        gen={'model': cfg.llm.endpoints[endpoint].model if endpoint else 'injected',
                             'prompt_version': 'eval-v1', 'created': datetime.now(timezone.utc).isoformat(),
                             'source_doc_ids': sorted({c['doc_id'] for c in selected})})
                    # 末轮答案不作为输入/历史；reference已承担此字段。
                    record['turns'][-1].pop('answer', None)
                    accepted.append(record); vectors.append(vector)
                    per_doc.update({c['doc_id'] for c in selected})
                    if on_accept is not None:
                        on_accept(record)
                    success = True
                    break
            except Exception as exc:
                rejects['backend_' + type(exc).__name__] += 1
            finally:
                async with changed:
                    reserved.subtract({c['doc_id'] for c in selected})
                    changed.notify_all()
            if not success:
                missing[kind] += 1

    # 混合类型轮转，文档名额不足时不会只留下先排的single题。
    jobs = []
    for index in range(max(settings.counts.values(), default=0)):
        for kind, count in settings.counts.items():
            if initial[kind] <= index < count:
                jobs.append(kind)
    if concurrency == 1:
        # 顺序模式保持原CLI/测试的type顺序。
        jobs = [k for k, n in settings.counts.items() for _ in range(max(0, n-initial[k]))]
        for kind in jobs:
            await one(kind)
    else:
        await asyncio.gather(*(one(kind) for kind in jobs))
    return accepted, {'requested': settings.counts, 'accepted': dict(Counter(r['type'] for r in accepted)),
                     'missing': dict(missing), 'rejections': dict(rejects), 'per_doc': dict(per_doc),
                     'resumed_records': len(resume_records or []), 'concurrency': concurrency}


async def _main(args):
    with metrics_scope(None), counted_calls(args.max_calls, ledger=Path(args.output or "eval/dataset.jsonl").with_suffix(".calls.json")):
        async with AsyncExitStack() as stack:
            return await _main_open(args, stack)


async def _main_open(args, stack):
    cfg = load_config(args.config)
    chunks = [json.loads(s) for s in Path(args.chunks).read_text().splitlines() if s.strip()]
    # 配额授权只决定可调用服务，不等于允许无限出题。CLI显式上限拦住重生成和fallback。
    calls = 0
    budget = args.max_calls
    async def bounded_call(scene, eps, schema, messages):
        nonlocal calls
        # structured_call 每次最多主备2个物理调用，保守预留2。
        if calls + 2 > budget:
            return None, None  # 保存部分结果与missing，不继续网络请求
        calls += 2
        return await structured_call(scene, eps, schema, messages)
    from eval.embeddings import LocalEmbeddings
    embeddings = LocalEmbeddings.from_config(cfg)
    deps = None
    if cfg.eval.generate.counts.get("unanswerable", 0):
        if args.deps_factory:
            from eval.run import _load_symbol
            deps = _load_symbol(args.deps_factory)(cfg, cfg.services.milvus.collection)
            if hasattr(deps, "__await__"):
                deps = await deps
        else:
            from eval.runtime import open_deps
            deps = await stack.enter_async_context(open_deps(cfg,cfg.services.milvus.collection))
        deps = replace(deps, structured_call=bounded_call, write=lambda event:None)
    from ragagent.llm.client import close_clients
    stack.push_async_callback(close_clients)
    async def retrieve(question):
        from ragagent.retrieval.pipeline import run_pipeline
        return await run_pipeline(question, deps)
    output = Path(args.output or cfg.eval.dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    resume_from = getattr(args, "resume_from", None)
    resumed = [json.loads(l) for l in Path(resume_from).read_text().splitlines() if l.strip()] if resume_from else []
    output.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in resumed))
    def save_accepted(record):
        with output.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"accepted={record['id']} type={record['type']} actual_calls={current_summary().get('actual_calls', 0)}", flush=True)
    records, stats = await generate(chunks, cfg, call=bounded_call, embed=embeddings.aembed_documents,
                                   retrieve=retrieve, seed=args.seed, on_accept=save_accepted,
                                   concurrency=getattr(args, "concurrency", 1), resume_records=resumed)
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    output.with_suffix(".generation.json").write_text(json.dumps({**stats, "reserved_physical_calls": calls, **current_summary()}, ensure_ascii=False, indent=2))
    print(f"generated={len(records)} missing={sum(stats['missing'].values())} output={output}")
    return 1 if stats["missing"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output")
    parser.add_argument("--deps-factory")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume-from", help="恢复已接受题；保留原id，继承文档名额和向量去重")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-calls", type=int, required=True, help="已获批准的物理LLM调用上限（保守预留主备两次）")
    args = parser.parse_args(argv)
    if args.max_calls < 1:
        parser.error("--max-calls must be positive")
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
