"""两档评测与回归门槛；真实缺失不能由夹具或缺失指标冒充通过。"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import math
import os
import sys
import time
import uuid
from collections import defaultdict
from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from eval.calls import counted_calls, current_summary
from contextlib import nullcontext
from eval.align import align_gold
from eval.metrics import ndcg_at_k, percentile, recall_at_k, reciprocal_rank
from ragagent.llm.observability import metrics_scope
from ragagent.config import load_config


def check_gates(current: dict, baseline: dict | None, gates, tier: str) -> list[dict]:
    """gate 值 None/NaN/基线缺失是 unverified（非成功）；百分点除以100。"""
    gates = gates.model_dump() if hasattr(gates, "model_dump") else gates
    specs = [("recall_at_8", "drop", gates["quick"]["recall_at_8_drop_pp"] / 100),
             ("chat_misroute", "max", gates["quick"]["chat_misroute_max"]),
             ("p95_latency_s", "rise", gates["quick"]["p95_latency_rise"])]
    if tier == "full":
        specs += [("faithfulness", "drop", gates["full"]["faithfulness_drop"]),
                  ("citation_support_rate", "drop", gates["full"]["citation_support_drop_pp"] / 100),
                  ("unanswerable_rate", "drop", gates["full"]["unanswerable_drop_pp"] / 100)]
    results = []
    for name, kind, threshold in specs:
        value, previous = current.get(name), (baseline or {}).get(name)
        status, delta = "unverified", None
        finite = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
        valid_value = finite(value) and value >= 0 and (name == "p95_latency_s" or value <= 1)
        valid_previous = finite(previous) and previous >= 0 and (name == "p95_latency_s" or previous <= 1)
        if valid_value and (kind == "max" or valid_previous):
            delta = value if kind == "max" else previous - value if kind == "drop" else (value / previous - 1 if previous > 0 else (0 if value == 0 else math.inf))
            # 消除恰好门槛值的浮点误差；超过才失败。
            status = "fail" if delta > threshold and not math.isclose(delta, threshold, abs_tol=1e-12) else "pass"
        results.append({"metric": name, "status": status, "threshold": threshold,
                        "current": value, "baseline": previous, "delta": delta if delta is None or math.isfinite(delta) else "infinite"})
    return results


def baseline_stability(first: dict, second: dict, gates) -> list[dict]:
    """两次完整档对称比较，下降/上升门槛任一方向超限都不能首次建基线。"""
    checks = check_gates(first, second, gates, "full") + check_gates(second, first, gates, "full")
    return checks


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _ids(rows):
    return [r["chunk_id"] if isinstance(r, dict) else r for r in rows]


def _summary(rows):
    fields = ("intent_accuracy", "chat_misroute", "recall_at_30", "recall_at_8", "mrr", "ndcg_at_8",
              "context_coverage", "scope_precision", "scope_recall", "scope_wrong_rate", "unanswerable_rate",
              "answerable_rejected", "faithfulness", "answer_relevancy", "context_precision", "context_recall",
              "answer_correctness", "citation_validity_rate", "citation_support_rate", "citation_coverage",
              "original_recall_at_8", "standalone_recall_at_8")
    result = {f: (None if any(f in r.get("required_metrics", []) and r.get(f) is None for r in rows)
                  else _mean([r[f] for r in rows if r.get(f) is not None])) for f in fields}
    latency = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    latency_missing = any("p95_latency_s" in r.get("required_metrics", []) and r.get("latency_s") is None for r in rows)
    result.update(p50_latency_s=percentile(latency, .5) if latency and not latency_missing else None,
                  p95_latency_s=percentile(latency, .95) if latency and not latency_missing else None,
                  budget_dropped_hits=sum(r.get("budget_dropped_hits", 0) for r in rows))
    return result


async def evaluate(dataset, chunks, backend, *, tier, cfg, collection, config_hash, mode="real", score=None, no_answer_judge=None, states_output=None):
    records, groups, stages = [], defaultdict(list), defaultdict(list)
    stale = 0
    for case in dataset:
        gold = align_gold(case.get("gold", []), chunks)
        if gold is None:
            stale += 1
            records.append({"id": case["id"], "status": "stale"})
            continue
        start = time.perf_counter()
        try:
            result = await backend(case, tier)
            if not isinstance(result, dict):
                raise TypeError("backend must return RAGState dictionary")
        except Exception as exc:
            row = {"id": case["id"], "status": "failed", "error_type": type(exc).__name__,
                   "required_metrics": ["recall_at_8", "chat_misroute", "p95_latency_s"]}
            if tier == "full":
                row["required_metrics"] += ["faithfulness", "citation_support_rate", "unanswerable_rate"]
            records.append(row)
            for group in {"all", case["type"], *case.get("tags", [])}:
                groups[group].append(row)
            continue
        elapsed = time.perf_counter() - start
        expected = case["expect"]
        intent = result.get("intent")
        row = {"id": case["id"], "status": "ok", "intent_accuracy": float(intent == expected["intent"]),
               "chat_misroute": float(intent == "chat") if expected["intent"] == "rag" else None,
               "required_metrics": []}
        rt = result.get("retrieval", {})
        row["latency_s"] = (rt.get("total_ms", elapsed * 1000) / 1000 if mode == "fixture" or tier == "quick" else result.get("eval_latency_s", elapsed))
        for stage, duration in rt.get("timings_ms", {}).items():
            stages[stage].append(duration / 1000)
        for stage, duration in result.get("timings", {}).items():
            stages[stage].append(duration)
        ranked, candidates = _ids(result.get("hits", [])), _ids(result.get("candidates", []))
        context = set(cid for b in result.get("context_blocks", []) for cid in b.get("chunk_ids", []))
        if gold:
            row.update(recall_at_30=recall_at_k(candidates, gold, 30), recall_at_8=recall_at_k(ranked, gold, 8),
                       mrr=reciprocal_rank(ranked, gold), ndcg_at_8=ndcg_at_k(ranked, gold, 8),
                       context_coverage=recall_at_k(list(context), gold, len(context)),
                       budget_dropped_hits=len(set(ranked) - context))
        predicted_scope = set((result.get("soft_filters") or {}).get("doc_ids", rt.get("scope_doc_ids", [])))
        expected_scope = set(expected.get("scope_doc_ids", []))
        if predicted_scope or expected_scope:
            overlap = len(predicted_scope & expected_scope)
            row.update(scope_precision=overlap / len(predicted_scope) if predicted_scope else 0,
                       scope_recall=overlap / len(expected_scope) if expected_scope else 0,
                       scope_wrong_rate=float(bool(predicted_scope - expected_scope)))
        if case["type"] == "unanswerable":
            handled = result.get("unanswerable_handled")
            if handled is None and (result.get("route", rt.get("route")) == "fallback" or result.get("gen_meta", {}).get("scene") == "fallback"):
                handled = True
            if handled is None and no_answer_judge is not None:
                try:
                    handled = await no_answer_judge(case, result)
                except Exception as exc:
                    row.update(status="partial", judge_error_type=type(exc).__name__)
            row["unanswerable_rate"] = float(handled) if handled is not None else None
        elif gold:
            row["answerable_rejected"] = float(result.get("route", rt.get("route")) == "fallback" or result.get("gen_meta", {}).get("scene") == "fallback")
        if case["type"] == "multi_turn" and gold:
            row["standalone_recall_at_8"] = row["recall_at_8"]
            if "original_hits" in result:
                row["original_recall_at_8"] = recall_at_k(_ids(result["original_hits"]), gold, 8)
        if tier == "full":
            post = result.get("gen_meta", {}).get("post", {})
            for key in ("citation_validity_rate", "citation_support_rate", "citation_coverage"):
                aliases = {"citation_validity_rate":"citation_validity", "citation_support_rate":"citation_support", "citation_coverage":"coverage"}
                row[key] = post.get(key, post.get(aliases[key]))
                if key == "citation_support_rate" and post.get("verify_degraded"):
                    row[key] = None
            if expected.get("answerable") and gold:
                row["required_metrics"] = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness",
                                           "citation_validity_rate", "citation_support_rate", "citation_coverage"]
                try:
                    row.update(await score(case, result) if score else result.get("ragas", {}))
                except Exception as exc:
                    row.update(status="partial", score_error_type=type(exc).__name__)
            if case["type"] == "unanswerable":
                row["required_metrics"].append("unanswerable_rate")
            meta = result.get("gen_meta", {})
            for stage, key in (("generate", "total_ms"), ("ttft", "ttft_ms")):
                if meta.get(key) is not None:
                    stages[stage].append(meta[key] / 1000)
            if post.get("verify_ms") is not None:
                stages["postprocess"].append(post["verify_ms"] / 1000)
        if states_output is not None:
            artifact = {"case":case,"state":{k:result.get(k) for k in ("answer","context_blocks","gen_meta")}}
            with Path(states_output).open("a") as stream:
                stream.write(json.dumps(artifact,ensure_ascii=False,allow_nan=False)+"\n")
        records.append(row)
        if mode == "real":
            print(f"evaluated={case['id']} status={row['status']} actual_calls={current_summary().get('actual_calls', 0)}", flush=True)
        for group in {"all", case["type"], *case.get("tags", [])}:
            groups[group].append(row)
    if not groups["all"]:
        raise ValueError("no evaluable questions (dataset empty or all stale)")
    ds_hash = hashlib.sha256(json.dumps(dataset, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"tier": tier, "mode": mode, "config_hash": config_hash, "prompt_version": cfg.prompt.version,
            "dataset_version": ds_hash, "collection": collection, "stale": stale,
            "models": {name: {"primary": cfg.llm.endpoints[scene.primary].model,
                              "fallback": cfg.llm.endpoints[scene.fallback].model}
                       for name in ("generate", "chat", "preprocess", "rewrite", "eval")
                       for scene in [getattr(cfg.llm.scenes, name)]},
            "metrics": _summary(groups["all"]), "groups": {k: _summary(v) for k, v in groups.items()},
            "stages": {k: {"p50": percentile(v, .5), "p95": percentile(v, .95)} for k, v in stages.items()},
            "cases": records}


def write_report(report, reports_dir):
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    path = root / (stem + ".json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    md = [f'# {report["tier"]} evaluation ({report["mode"]})', f'Stale: {report["stale"]}',
          "", "| Metric | Current | Baseline | Gate |", "|---|---:|---:|---|"]
    for gate in report.get("gates", []):
        status = "PASS" if gate["status"] == "pass" else "🔴 FAIL" if gate["status"] == "fail" else "UNVERIFIED"
        md.append(f'| {gate["metric"]} | {gate["current"]} | {gate["baseline"]} | {status} |')
    if report["mode"] == "fixture":
        md += ["", "Fixture shell only: NOT a real M4 evaluation or production baseline."]
    path.with_suffix(".md").write_text("\n".join(md) + "\n")
    return path


def _load_symbol(path):
    module, name = path.split(":", 1)
    return getattr(importlib.import_module(module), name)


def _messages(case):
    from langchain_core.messages import AIMessage, HumanMessage
    history = []
    for turn in case["turns"][:-1]:
        history.append(HumanMessage(content=turn["question"]))
        if "answer" in turn:
            history.append(AIMessage(content=turn["answer"]))
    return history


def real_backend(deps):
    graph = None
    async def backend(case, tier):
        nonlocal graph
        from ragagent.retrieval.pipeline import run_original, run_pipeline
        question, messages = case["turns"][-1]["question"], _messages(case)
        started = time.perf_counter()
        if tier == "quick":
            result = await run_pipeline(question, deps, filters=None, messages=messages)
        else:
            from ragagent.graph.build import build_graph
            if graph is None:
                graph = build_graph()  # checkpointer不接入真实用户历史
            result = await graph.ainvoke({"question":question,"messages":messages,"filters":None},
                context=deps, config={"callbacks":[],"configurable":{"thread_id":"l4-eval-"+uuid.uuid4().hex}})
        result["eval_latency_s"] = time.perf_counter() - started
        if case["type"] == "multi_turn":
            original = await run_original(question,deps,filters=None,messages=messages)
            result["original_hits"] = original.get("hits", [])
        return result
    return backend


def validate_dataset(dataset):
    from eval.generate import GeneratedQuestion, TYPES
    seen = set()
    for case in dataset:
        if not isinstance(case.get("id"), str) or not case["id"] or case["id"] in seen:
            raise ValueError("dataset ids must be unique nonempty strings")
        seen.add(case["id"])
        if case.get("type") not in TYPES:
            raise ValueError(f"unknown dataset type: {case.get('type')}")
        q = GeneratedQuestion.model_validate({k:case[k] for k in ("tags","turns","expect","gold","reference")})
        answered = case["type"] not in {"chat","clarify","unanswerable"}
        intent = case["type"] if case["type"] in {"chat","clarify"} else "rag"
        if q.expect.answerable != answered or q.expect.intent != intent:
            raise ValueError("dataset expect/type mismatch")
        if answered and (not q.gold or not q.reference.strip()):
            raise ValueError("answerable dataset questions require quote gold and reference")
        if not answered and q.gold:
            raise ValueError("unanswerable/chat/clarify questions must not have gold")
        if case["type"] == "multi_turn" and (len(q.turns)<2 or not q.expect.standalone or any(not t.answer for t in q.turns[:-1])):
            raise ValueError("multi_turn dataset needs prior answers and standalone")
    if not dataset:
        raise ValueError("empty dataset")


async def _run(args):
    if not args.fixture and not getattr(args, 'max_calls', None):
        raise ValueError('real evaluation requires explicit --max-calls for all LLM requests')
    counter = nullcontext() if args.fixture else counted_calls(args.max_calls, ledger=args.calls_output)
    with metrics_scope(None), counter:
        async with AsyncExitStack() as stack:
            return await _run_open(args, stack)


async def _run_open(args, stack):
    raw = yaml.safe_load(Path(args.config).read_text())
    env = os.environ.copy()
    if args.fixture:
        for ep in raw["llm"]["endpoints"].values():
            env.setdefault(ep["api_key_env"], "fixture-not-used")
    cfg = load_config(args.config, env=env)
    if args.fixture:
        from eval.fixtures import FixtureBackend, fixture_chunks, fixture_dataset
        dataset, chunks, backend = fixture_dataset(), fixture_chunks(), FixtureBackend(args.fixture_regression)
    else:

        dataset = [json.loads(line) for line in Path(args.dataset or cfg.eval.dataset).read_text().splitlines() if line.strip()]
        validate_dataset(dataset)
        files = [Path(args.chunks)] if args.chunks else sorted(Path(cfg.paths.chunks).rglob("*.jsonl"))
        if not files:
            raise ValueError("no current chunks; provide --chunks current.jsonl or finish M1b")
        chunks = [json.loads(line) for file in files for line in file.read_text().splitlines() if line.strip()]
        collection = args.collection or cfg.services.milvus.collection
        cfg = cfg.model_copy(update={"services":cfg.services.model_copy(update={
            "milvus":cfg.services.milvus.model_copy(update={"collection":collection})})})
        if args.deps_factory:
            deps = _load_symbol(args.deps_factory)(cfg, collection)
            deps = await deps if inspect.isawaitable(deps) else deps
            from ragagent.llm.client import close_clients
            stack.push_async_callback(close_clients)
        else:
            from eval.runtime import open_deps
            deps = await stack.enter_async_context(open_deps(cfg,collection))
        from eval.cache import PreprocessCache
        cache = PreprocessCache(cfg, deps.structured_call, Path(cfg.eval.reports_dir) / "cache" / "preprocess")
        deps = replace(deps, cfg=cfg, structured_call=cache, write=lambda event:None)
        backend = real_backend(deps)
    mode = "fixture" if args.fixture else "real"
    collection = "l4_fixture" if args.fixture else args.collection or cfg.services.milvus.collection
    config_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    kwargs = dict(tier=args.tier, cfg=cfg, collection=collection, config_hash=config_hash, mode=mode)
    if not args.fixture and args.tier == "full":
        if args.max_eval_calls is None:
            raise ValueError("real full evaluation requires explicit --max-eval-calls for judge requests")
        from eval.ragas_eval import RagasScorer
        scorer = RagasScorer(cfg, max_calls=args.max_eval_calls, prompt_dir=args.ragas_prompt_dir)
        kwargs.update(score=scorer, no_answer_judge=scorer.judge_no_answer)
    if args.states_output:
        Path(args.states_output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.states_output).write_text("")
        kwargs["states_output"] = args.states_output
    report = await evaluate(dataset, chunks, backend, **kwargs)
    baseline_dir = Path(cfg.eval.baselines_dir) / "fixtures" if args.fixture else Path(cfg.eval.baselines_dir)
    baseline_path = baseline_dir / f"{args.tier}.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    if args.update_baseline:
        report["gates"] = check_gates(report["metrics"], report["metrics"], cfg.eval.gates, args.tier)
        if args.tier == "full":
            second = await evaluate(dataset, chunks, backend, **kwargs)
            report["baseline_stability"] = baseline_stability(report["metrics"], second["metrics"], cfg.eval.gates)
            report["baseline_stability_mode"] = mode
            report["second_run"] = second
        else:
            report["baseline_stability"] = []
        accepted = all(g["status"] == "pass" for g in report["gates"] + report["baseline_stability"])
        if accepted:
            baseline_dir.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        report["baseline_updated"] = accepted
    else:
        compatible = baseline and baseline.get("tier") == args.tier and baseline.get("mode") == mode and baseline.get("dataset_version") == report["dataset_version"]
        report["gates"] = check_gates(report["metrics"], baseline.get("metrics") if compatible else None, cfg.eval.gates, args.tier)
        if not compatible:
            report["baseline_error"] = "missing or incompatible baseline; use --update-baseline only after approval"
        accepted = all(g["status"] == "pass" for g in report["gates"])
    if not args.fixture:
        report.update(current_summary())
        report["failed"] = sum(r.get("status") in {"partial", "failed"} for r in report["cases"])
        report["preprocess_cache"] = {"hits":cache.hits,"misses":cache.misses}
        if args.tier == "full":
            report["eval_reserved_calls"] = scorer.budget.used
    path = write_report(report, Path(cfg.eval.reports_dir) / "fixtures" if args.fixture else cfg.eval.reports_dir)
    print(f"report={path} mode={mode} tier={args.tier} stale={report['stale']}")
    for gate in report["gates"]:
        print(f"{gate['metric']}: {gate['status']} current={gate['current']} baseline={gate['baseline']}")
    return 0 if accepted else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["quick", "full"], required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--collection")
    parser.add_argument("--chunks")
    parser.add_argument("--deps-factory")
    parser.add_argument("--max-calls", type=int, help="全链实际HTTP调用上限，真实档必须显式设置")
    parser.add_argument("--calls-output", default="eval/reports/run.calls.json")
    parser.add_argument("--max-eval-calls", type=int, help="真实完整档评测模型调用上限；主备保守计数")
    parser.add_argument("--ragas-prompt-dir")
    parser.add_argument("--states-output", help="仅本地保存实际回答/上下文，供中文30条验证")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--fixture-regression", action="store_true")
    args = parser.parse_args(argv)
    if args.fixture_regression and not args.fixture:
        parser.error("--fixture-regression requires --fixture")
    try:
        return asyncio.run(_run(args))
    except Exception as exc:
        print(f"evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
