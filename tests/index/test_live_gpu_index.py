"""显式 L1_REAL_INDEX=1 时验证真实GPU→sealed DISKANN→图片/增量。

L1_REAL_INDEX=1 uv run --env-file .env pytest -s tests/index/test_live_gpu_index.py
仅自生成合成MD/PNG；独立l1_集合与登记表，不改真实语料或正式别名。
"""
import json
import os
from pathlib import Path
import shutil
import uuid

from PIL import Image, ImageDraw
import pytest

from ragagent.config import load_config
from ragagent.index.embedder import Embedder
from ragagent.index.milvus import connect, load
from ragagent.index.pipeline import run_index
from ragagent.ingest.chunking.pipeline import run_chunk
from ragagent.ingest.models import CleanDocument
from ragagent.ingest.pipeline import run_clean
from ragagent.ingest.registry_db import RegistryDB


@pytest.mark.skipif(os.environ.get('L1_REAL_INDEX') != '1', reason='需显式启用真实GPU和Milvus联合验收')
def test_real_gpu_image_and_incremental_pipeline(tmp_path):
    token = uuid.uuid4().hex[:12]
    coll = 'l1_gpu_index_' + token
    cfg = load_config()
    cfg = cfg.model_copy(update={
        'paths': cfg.paths.model_copy(update={'registry_db': str(tmp_path / 'registry.db'),
            'chunks': str(tmp_path / 'chunks'), 'parsed': str(tmp_path / 'parsed')}),
        'chunking': cfg.chunking.model_copy(update={'target_tokens': 64, 'max_tokens': 120,
            'min_tokens': 12, 'overlap_tokens': 12})})
    raw = tmp_path / 'raw'
    raw.mkdir()
    source = raw / ('validation-' + token + '.md')
    picture = raw / 'red-square.png'
    image = Image.new('RGB', (448, 448), 'white')
    ImageDraw.Draw(image).rectangle((100, 100, 340, 340), fill='red')
    image.save(picture)
    source.write_text(''.join(
        f'# Validation T{i:05d}\n\nError code E{i:05d}: ' + 'Check connection and deployment configuration. ' * 14 + '\n\n'
        for i in range(1056)) + '# Red square figure\n\n![A red square on a white background](red-square.png)\n')
    client = connect(cfg)
    owned_clean = []
    try:
        with RegistryDB(cfg.paths.registry_db) as db, Embedder(cfg=cfg, timeout=300) as emb:
            health = emb.health()
            assert health['device'] == 'cuda' and health['model'] == 'visualized-bge-m3'
            assert run_clean(raw, cfg=cfg, registry=db, local_only=True)['cleaned'] == 1
            record = db.list_documents()[0]
            owned_clean.append(Path(cfg.paths.clean) / record['doc_id'])

            def chunk_current():
                current = db.get_document(record['doc_id'])
                doc = CleanDocument.model_validate_json((Path(cfg.paths.clean) / current['doc_id'] /
                    f'v{current["version"]}' / 'clean.json').read_text())
                chunks = run_chunk(doc, cfg=cfg.chunking, encode=emb.text)
                output = Path(cfg.paths.chunks) / doc.doc_id / f'v{doc.version}.jsonl'
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(''.join(c.model_dump_json() + '\n' for c in chunks))
                return doc, chunks

            doc, chunks = chunk_current()
            assert len(chunks) > 1024  # 满足真实Milvus索引构建的segment阈值
            pictures = [c for c in chunks if c.type == 'image']
            assert len(pictures) == 1 and pictures[0].image_path.startswith('clean/')
            indexed = run_index(cfg=cfg, client=client, emb=emb, registry=db, collection=coll)
            assert not indexed['errors'] and indexed['chunks'] == len(chunks)
            assert indexed['indexes']['dense']['index_type'] == 'DISKANN'
            assert all(v['indexed_rows'] == v['total_rows'] > 0 for v in indexed['indexes'].values())
            hit = client.search(coll, data=emb.text(['A red square on a white background']), anns_field='dense',
                search_params={'metric_type': 'COSINE', 'params': {'search_list': 100}}, limit=10,
                consistency_level='Strong')[0]
            assert pictures[0].chunk_id in [h['id'] for h in hit]
            print('REAL_GPU_IMAGE_TOP10', pictures[0].chunk_id)

            source.write_text('# Updated deployment\n\nNew version EUPDATE uses config.yaml for deployment. ' * 8)
            assert run_clean(raw, cfg=cfg, registry=db, local_only=True)['cleaned'] == 1
            updated, parts = chunk_current()
            assert updated.version == 2
            report = run_index(cfg=cfg, client=client, emb=emb, registry=db, collection=coll)
            assert not report['errors']
            again = run_index(cfg=cfg, client=client, emb=emb, registry=db, collection=coll)
            assert again['documents'] == 0 and again['skipped'] == 1
            load(client, coll)
            expr = f'doc_id == "{doc.doc_id}"'
            count = client.query(coll, filter=expr, output_fields=['count(*)'], consistency_level='Strong')[0]['count(*)']
            old = client.query(coll, filter=expr + ' and version < 2', output_fields=['count(*)'], consistency_level='Strong')[0]['count(*)']
            assert count == len(parts) and old == 0
            assert db.get_document(doc.doc_id)['status'] == 'indexed'
            print('REAL_GPU_INCREMENTAL_OK', json.dumps({'new': count, 'old': old, 'idempotent': True}))
    finally:
        if client.has_collection(coll):
            client.drop_collection(coll)
        client.close()
        for folder in owned_clean:
            if folder.is_dir() and not folder.is_symlink():
                shutil.rmtree(folder)
