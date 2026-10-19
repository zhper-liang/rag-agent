import multiprocessing
from pathlib import Path
import sys
import pytest
from ragagent.ingest.lock import IngestBusy, ingest_lock


def contender(path, queue):
    try:
        with ingest_lock(path):
            queue.put('acquired')
    except IngestBusy:
        queue.put('busy')


def test_two_process_contention_and_release(tmp_path):
    path = tmp_path / 'ingest.lock'
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    with ingest_lock(path):
        process = ctx.Process(target=contender, args=(path, queue))
        process.start()
        assert queue.get(timeout=10) == 'busy'
        process.join(10)
        assert process.exitcode == 0
    process = ctx.Process(target=contender, args=(path, queue))
    process.start()
    assert queue.get(timeout=10) == 'acquired'
    process.join(10)
    assert process.exitcode == 0
    assert path.is_file()  # 不 unlink，避免 inode 竞争


@pytest.mark.parametrize('module,args', [
    ('ragagent.ingest.pipeline', []),
    ('ragagent.index.pipeline', []),
    ('ragagent.ingest.chunking.pipeline', ['absent.json']),
])
def test_cli_takes_lock_before_side_effects(tmp_path, monkeypatch, module, args):
    import importlib
    from ragagent.config import load_config
    from tests.integration.helpers import KEYS
    mod = importlib.import_module(module)
    cfg = load_config(env={k:'test' for k in KEYS})
    path = tmp_path / 'ingest.lock'
    cfg = cfg.model_copy(update={'paths': cfg.paths.model_copy(update={'ingest_lock': str(path)})})
    monkeypatch.setattr(mod, 'load_config', lambda *a, **kw: cfg)
    monkeypatch.setattr(sys, 'argv', ['entry', *args])
    with ingest_lock(path), pytest.raises(SystemExit, match='已有入库任务'):
        mod.main()
