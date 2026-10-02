import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from eval.fixtures import FixtureBackend, fixture_chunks, fixture_dataset
from eval.run import baseline_stability, check_gates, evaluate


def gates():
    return yaml.safe_load(Path("config.yaml").read_text())["eval"]["gates"]


def metrics():
    return dict(recall_at_8=.8, chat_misroute=.01, p95_latency_s=1, faithfulness=.9,
                citation_support_rate=.95, unanswerable_rate=.9)


def test_all_gates_equal_boundary_worse_and_missing():
    base = metrics()
    assert all(g["status"] == "pass" for g in check_gates(base, base, gates(), "full"))
    boundary = dict(recall_at_8=.78, chat_misroute=.02, p95_latency_s=1.2, faithfulness=.87,
                    citation_support_rate=.92, unanswerable_rate=.85)
    assert all(g["status"] == "pass" for g in check_gates(boundary, base, gates(), "full"))
    worse = dict(recall_at_8=.77, chat_misroute=.021, p95_latency_s=1.21, faithfulness=.869,
                 citation_support_rate=.919, unanswerable_rate=.849)
    assert all(g["status"] == "fail" for g in check_gates(worse, base, gates(), "full"))
    assert check_gates({}, None, gates(), "quick")[0]["status"] == "unverified"
    assert check_gates({**base, "faithfulness": float("nan")}, base, gates(), "full")[3]["status"] == "unverified"
    assert check_gates({**base, "p95_latency_s": 1}, {**base, "p95_latency_s": 0}, gates(), "quick")[2]["status"] == "fail"
    assert any(g["status"] == "fail" for g in baseline_stability(base, {**base, "faithfulness": .7}, gates()))


def test_cli_quick_full_equal_bad_and_explicit_baselines(tmp_path):
    raw = yaml.safe_load(Path("config.yaml").read_text())
    raw["eval"]["baselines_dir"] = str(tmp_path / "baselines")
    raw["eval"]["reports_dir"] = str(tmp_path / "reports")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw))
    def run(*args):
        return subprocess.run([sys.executable, "-m", "eval.run", "--config", str(config), "--fixture", *args],
                              capture_output=True, text=True, timeout=30)
    for tier in ("quick", "full"):
        missing = run("--tier", tier)
        assert missing.returncode == 1
        assert not (tmp_path / "baselines" / "fixtures" / f"{tier}.json").exists()
        initial = run("--tier", tier, "--update-baseline")
        assert initial.returncode == 0, initial.stderr
        path = tmp_path / "baselines" / "fixtures" / f"{tier}.json"
        before = path.read_bytes()
        assert run("--tier", tier).returncode == 0
        failed = run("--tier", tier, "--fixture-regression")
        assert failed.returncode == 1, failed.stderr
        assert "recall_at_8: fail" in failed.stdout
        assert path.read_bytes() == before
        baseline = json.loads(before)
        assert baseline["mode"] == "fixture" and baseline["stale"] == 1
        if tier == "full":
            assert baseline["baseline_stability_mode"] == "fixture"
        assert not (tmp_path / "baselines" / f"{tier}.json").exists()
    reports = list((tmp_path / "reports" / "fixtures").glob("*.json"))
    assert len(reports) == 8
    assert all(p.with_suffix(".md").exists() for p in reports)
    assert any("🔴 FAIL" in p.with_suffix(".md").read_text() for p in reports)


def test_dataset_with_all_stale_rejected():
    # stale 的整题不进分母；不能把空集伪造为完美Recall。
    case = fixture_dataset()[-1]
    cfg = SimpleNamespace()
    with pytest.raises(ValueError, match="no evaluable"):
        asyncio.run(evaluate([case], fixture_chunks(), FixtureBackend(), tier="quick", cfg=cfg,
                             collection="l4_fixture", config_hash="hash", mode="fixture"))


def test_required_metric_missing_in_one_case_cannot_be_ignored():
    from eval.run import _summary
    result=_summary([{'faithfulness':1.,'required_metrics':['faithfulness']},
                     {'faithfulness':None,'required_metrics':['faithfulness']}])
    assert result['faithfulness'] is None
    assert check_gates(result,metrics(),gates(),'full')[3]['status']=='unverified'


def test_cache_key_history_prompt_models_and_nonpreprocess(tmp_path):
    from eval.cache import PreprocessCache
    from ragagent.config import Config
    from ragagent.preprocess.schema import Preprocessed
    from langchain_core.messages import HumanMessage
    cfg=Config.model_validate(yaml.safe_load(Path('config.yaml').read_text()))
    raw=json.loads(Path('tests/fixtures/preprocess_out.json').read_text())[0]
    calls=[]
    async def call(scene,eps,schema,messages):
        calls.append(1)
        return schema.model_validate(raw),scene.primary
    async def run():
        cache=PreprocessCache(cfg,call,tmp_path)
        args=(cfg.llm.scenes.preprocess,cfg.llm.endpoints,Preprocessed)
        messages=[HumanMessage('合成测试问题')]
        await cache(*args,messages)
        await cache(*args,messages)
        assert len(calls)==1 and cache.hits==1
        await cache(*args,[HumanMessage('不同历史'),*messages])
        cfg2=cfg.model_copy(update={'prompt':cfg.prompt.model_copy(update={'version':'different'})})
        await PreprocessCache(cfg2,call,tmp_path)(*args,messages)
        changed={n:e.model_copy(update={'model':e.model+'-new'}) for n,e in cfg.llm.endpoints.items()}
        await cache(cfg.llm.scenes.preprocess,changed,Preprocessed,messages)
        assert len(calls)==4
    asyncio.run(run())
    assert len(list(tmp_path.glob('*.json')))==4


def test_actual_adapter_uses_graph_context_and_original_sidecar(monkeypatch):
    import ragagent.retrieval.pipeline as pipeline
    import ragagent.graph.build as graphmodule
    from eval.run import real_backend
    deps=object();seen=[]
    async def quick(q,d,**kw):
        assert d is deps
        seen.append(('quick',q,kw)); return {'hits':[{'chunk_id':'rewritten'}]}
    async def original(q,d,**kw):
        assert d is deps
        seen.append(('original',q,kw)); return {'hits':[{'chunk_id':'original'}]}
    class Graph:
        async def ainvoke(self,state,**kw):
            assert kw['context'] is deps and kw['config']['callbacks']==[]
            assert kw['config']['configurable']['thread_id'].startswith('l4-eval-')
            seen.append(('full',state,kw)); return {'hits':[{'chunk_id':'rewritten'}]}
    monkeypatch.setattr(pipeline,'run_pipeline',quick)
    monkeypatch.setattr(pipeline,'run_original',original)
    monkeypatch.setattr(graphmodule,'build_graph',lambda:Graph())
    case=fixture_dataset()[2]
    async def run():
        backend=real_backend(deps)
        for tier in ('quick','full'):
            r=await backend(case,tier)
            assert r['original_hits']==[{'chunk_id':'original'}]
            assert r['eval_latency_s']>=0
    asyncio.run(run())
    assert [x[0] for x in seen]==['quick','original','full','original']
    assert len(seen[0][2]['messages'])==2


def test_concurrent_evaluation_keeps_failed_case_and_input_order():
    from ragagent.config import Config
    cfg = Config.model_validate(yaml.safe_load(Path('config.yaml').read_text()))
    cases = fixture_dataset()[:3]
    active = peak = 0
    fixture = FixtureBackend()
    async def backend(case, tier):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(.01 if case is cases[0] else 0)
            if case is cases[1]:
                raise RuntimeError('backend unavailable')
            return await fixture(case, tier)
        finally:
            active -= 1
    report = asyncio.run(evaluate(cases, fixture_chunks(), backend, tier='quick', cfg=cfg,
                                  collection='l4_fixture', config_hash='x', mode='fixture', concurrency=2))
    assert peak == 2
    assert [r['id'] for r in report['cases']] == [c['id'] for c in cases]
    assert report['cases'][1]['status'] == 'failed'
    assert report['metrics']['recall_at_8'] is None
