import json
import logging
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
from ragagent.monitor.backup import backup_databases
from ragagent.monitor.samples import PromptSampler
from ragagent.monitor.gpu import gpu_metrics
from ragagent.prompt.models import PromptPackage, Message, TextPart
from tests.integration.helpers import config


def test_backup_wal_rotation_and_missing_source(tmp_path):
    cfg = config(tmp_path)
    cfg = cfg.model_copy(update={'backup': cfg.backup.model_copy(update={'dir':str(tmp_path/'backups'), 'keep':2})})
    db = sqlite3.connect(cfg.paths.registry_db)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE secret (n INTEGER)')
    db.execute('INSERT INTO secret VALUES (42)'); db.commit()
    for _ in range(3):
        report = backup_databases(cfg)
        assert report['errors'] == {}
        assert set(report['skipped']) == {'feedback_db', 'session_db'}
        with sqlite3.connect(report['backups']['registry_db']) as saved:
            assert saved.execute('select n from secret').fetchone() == (42,)
        assert Path(report['backups']['registry_db']).stat().st_mode & 0o777 == 0o600
    assert len(list((tmp_path/'backups').glob('registry_db-*.sqlite3'))) == 2
    assert not list((tmp_path/'backups').glob('*.partial'))
    assert not Path(cfg.paths.feedback_db).exists()
    db.close()


def test_backup_failure_keeps_previous_snapshots(tmp_path):
    cfg = config(tmp_path)
    cfg = cfg.model_copy(update={'backup':cfg.backup.model_copy(update={'dir':str(tmp_path/'out')})})
    Path(cfg.paths.registry_db).write_text('corrupt')
    report = backup_databases(cfg)
    assert report['errors']['registry_db'] == 'DatabaseError'
    assert not list((tmp_path/'out').glob('*.partial'))
    assert not list((tmp_path/'out').glob('*.sqlite3'))


def package(text='secret body'):
    return PromptPackage(scene='chat', system='secret system', version='v1', est_input_tokens=2,
                         messages=[Message(role='user', parts=[TextPart(text=text)])])


def sampler_cfg(tmp_path, rate):
    cfg = config(tmp_path)
    return cfg.model_copy(update={'logging':cfg.logging.model_copy(update={'prompt_sample_rate':rate,
        'sample_path':str(tmp_path/'samples'/'prompt.jsonl'), 'sample_max_mb':1, 'sample_keep_files':2})})


def test_sampling_disabled_creates_nothing(tmp_path):
    sampler = PromptSampler(sampler_cfg(tmp_path, 0))
    assert sampler.record(package(), 'answer', {'status':'ok'}) is False
    assert not (tmp_path/'samples').exists()


def test_samples_local_full_content_rotation_and_no_logging(tmp_path, caplog):
    cfg = sampler_cfg(tmp_path, 1)
    sampler = PromptSampler(cfg)
    for _ in range(4):
        assert sampler.record(package('资料'*130000), 'secret answer', {'status':'ok'}) is True
    sampler.close()
    path = Path(cfg.logging.sample_path)
    files = list(path.parent.glob('prompt.jsonl*'))
    assert 2 <= len(files) <= 3
    for file in files:
        assert file.stat().st_mode & 0o777 == 0o600
        for line in file.read_text().splitlines():
            row = json.loads(line)
            assert row['prompt']['system'] == 'secret system'
            assert row['answer'] == 'secret answer'
    assert 'secret' not in caplog.text


def test_sampling_rate_and_io_failure_never_leak(tmp_path, monkeypatch, caplog):
    sampler = PromptSampler(sampler_cfg(tmp_path, .5))
    monkeypatch.setattr('ragagent.monitor.samples.random.random', lambda: .6)
    assert not sampler.record(package(), 'secret answer', {'status':'ok'})
    monkeypatch.setattr('ragagent.monitor.samples.random.random', lambda: .4)
    monkeypatch.setattr('ragagent.monitor.samples.PrivateRotatingHandler.emit',
                        lambda *a: (_ for _ in ()).throw(OSError('secret answer')))
    assert not sampler.record(package(), 'secret answer', {'status':'ok'})
    assert 'secret answer' not in caplog.text
    sampler.close()


def test_gpu_metric_names_match_rules(monkeypatch):
    from scripts.gen_alerts import generate_rules
    from ragagent.config import load_config
    from tests.integration.helpers import KEYS
    monkeypatch.setattr('ragagent.monitor.gpu.subprocess.run',
                        lambda *a, **kw: SimpleNamespace(stdout='0, 1024, 8192\n'))
    text = gpu_metrics(2).decode()
    assert 'rag_gpu_memory_used_bytes{gpu="0"} 1.073741824e+09' in text
    rules = generate_rules(load_config(env={k:'test' for k in KEYS}).monitor.alerts)
    expr = next(r['expr'] for r in rules['groups'][0]['rules'] if r['alert']=='RagGpuMemRatio')
    assert 'rag_gpu_memory_used_bytes' in expr and 'rag_gpu_memory_total_bytes' in expr
    assert 'DCGM' not in expr


@pytest.mark.parametrize('failed,interrupted', [(False,False), (False,True), (True,False)])
def test_generate_node_calls_local_sampler_only(tmp_path, monkeypatch, failed, interrupted):
    import asyncio
    from langgraph.runtime import Runtime
    from ragagent.graph.deps import Deps
    from ragagent.graph.nodes import generate
    from ragagent.llm.stream import StreamResult
    from ragagent.llm.errors import GenerationFailed
    cfg = sampler_cfg(tmp_path, 1)
    sampler = PromptSampler(cfg)
    prompt = package()
    monkeypatch.setattr(generate, 'assemble', lambda *a, **kw: prompt)
    async def stream(*a):
        if failed:
            raise GenerationFailed()
        return StreamResult(text='private answer', finish_reason='stop', usage=None, ttft_ms=1,
                            interrupted=interrupted), {'endpoint':'main', 'fallback_used':False, 'images_dropped':0}
    monkeypatch.setattr(generate, 'stream_answer', stream)
    deps = Deps(cfg=cfg, count=len, write=lambda e: None, prompt_sampler=sampler)
    result = asyncio.run(generate.generate_node({}, Runtime(context=deps)))
    sampler.close()
    row = json.loads(Path(cfg.logging.sample_path).read_text())
    assert row['answer'] == ('' if failed else 'private answer')
    assert row['status'] == ('failed' if failed else 'interrupted' if interrupted else 'ok')
    assert result['gen_meta']['status'] == row['status']
    assert 'secret system' not in json.dumps(result)
    assert 'prompt_sampler' not in result
