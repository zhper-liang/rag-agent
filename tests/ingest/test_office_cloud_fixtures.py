"""授权合成 Office 的真实云端返回；所有测试必须完全离线。"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import socket
import zipfile

import pytest

from ragagent.config import load_config
from ragagent.ingest.dedup import sha256
from ragagent.ingest.mineru.client import extract_zip
from ragagent.ingest.mineru.convert import content_list_to_blocks
from ragagent.ingest.models import CleanDocument
from ragagent.ingest.pipeline import run_clean
from ragagent.ingest.registry_db import RegistryDB, make_doc_id

FIXTURES = Path(__file__).parent / 'fixtures' / 'mineru_office_cloud'
CASES = json.loads((FIXTURES / 'manifest.json').read_text())


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('云端夹具回归禁止网络调用')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['case'])
def test_real_office_return_complete_block_snapshot(case):
    raw = (FIXTURES / case['content_list']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == case['content_list_sha256']
    assert case['synthetic_only'] is True
    items = json.loads(raw)
    blocks = content_list_to_blocks(items)
    expected = json.loads((FIXTURES / case['expected_blocks']).read_text())
    assert [b.model_dump() for b in blocks] == expected
    assert dict(Counter(b.type for b in blocks)) == case['types']
    assert max(b.page for b in blocks) == case['pages']
    assert all(b.image_path is None for b in blocks)
    assert not any(b.type == 'equation' for b in blocks)
    number = Path(case['source_file']).stem.rsplit('-', 1)[1]
    if case['source_file'].endswith('.docx'):
        assert [b.type for b in blocks] == ['heading', 'heading', 'paragraph', 'table', 'paragraph', 'paragraph']
        assert [(b.text, b.level) for b in blocks[:2]] == [
            (f'**L1 Synthetic Word {number}**', 1), ('**Deployment procedure**', 2)]
        table = blocks[3]
        assert table.text == '|  |\n| --- |\n| Model |\n| BGE-M3 |\n| Reranker |'
        assert table.meta['columns'] == 1 and table.meta['has_header'] is False
        assert all(b.page == 1 for b in blocks)
    else:
        assert all(b.type == 'paragraph' and b.level == 0 for b in blocks)
        assert [b.page for b in blocks] == [1]*4 + [2]*4
        assert blocks[0].text == f'L1 Synthetic PPT {number} slide 1'
        assert blocks[4].text == f'L1 Synthetic PPT {number} slide 2'
        assert all('text_level' not in item for item in items)
    assert any('1024' in b.text and 'E00042' in b.text for b in blocks)
    assert any('中文' in b.text for b in blocks)
    assert any('Formula' in b.text for b in blocks)


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['case'])
def test_cached_cloud_return_runs_complete_clean_and_incremental(tmp_path, case):
    root = tmp_path / 'raw'
    root.mkdir()
    source = root / case['source_file']
    # 显式离线替身，不把测试字节当作真正 Office 文件或新的云端证据。
    source.write_bytes(b'OFFLINE synthetic source wrapper for cached response')
    doc_id = make_doc_id(source.name)
    parsed, clean = tmp_path / 'parsed', tmp_path / 'clean'
    cfg = load_config()
    cfg = cfg.model_copy(update={
        'paths': cfg.paths.model_copy(update={'raw': str(root), 'parsed': str(parsed),
            'clean': str(clean), 'registry_db': str(tmp_path/'registry.db')}),
        'ingest': cfg.ingest.model_copy(update={
            'mineru': cfg.ingest.mineru.model_copy(update={'model_version': case['model']})}),
    })
    archive = tmp_path / 'response.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('synthetic_content_list.json', (FIXTURES / case['content_list']).read_bytes())
        z.writestr('full.md', 'OFFLINE full.md wrapper; conversion reads content_list')
    extract_zip(archive, parsed/doc_id/'v1', source_hash=sha256(source))

    class NeverParse:
        async def parse_batch(self, *args, **kwargs):
            raise AssertionError('已完整校验缓存的解析产物不应调用云端客户端')

    with RegistryDB(cfg.paths.registry_db) as db:
        db.register(source.name, sha256(source), file_type=source.suffix[1:],
            parser='mineru-'+case['model'])
        db.update(doc_id, status='parsing', mineru_batch_id='offline-known-batch')
        first = run_clean(root, cfg=cfg, registry=db, mineru=NeverParse())
        assert first['cleaned'] == 1 and first['failed'] == 0
        record = db.get(doc_id)
        assert record['status'] == 'cleaned' and record['version'] == 1
        assert record['pages'] == case['pages'] and record['error'] is None
        assert record['mineru_batch_id'] == 'offline-known-batch'
        output = clean/doc_id/'v1'
        doc = CleanDocument.model_validate_json((output/'clean.json').read_text())
        assert doc.content_hash == sha256(source)
        assert doc.parser == 'mineru-'+case['model'] and doc.blocks
        hashes = {p.relative_to(output).as_posix(): sha256(p) for p in output.rglob('*') if p.is_file()}
        second = run_clean(root, cfg=cfg, registry=db, mineru=NeverParse())
        assert second['skipped'] == 1 and second['cleaned'] == second['failed'] == 0
        assert db.get(doc_id) == record
        assert hashes == {p.relative_to(output).as_posix(): sha256(p) for p in output.rglob('*') if p.is_file()}
