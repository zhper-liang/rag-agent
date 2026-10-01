"""L1_MILVUS=1 uv run --env-file .env pytest -s tests/index/test_m1a.py

真实HTTP桩→切分JSONL→登记表→writer→sealed DISKANN→hybrid search。
只创建/删除本次唯一l1_集合，不操作容器或rag_chunks别名。
"""
import json
import os
from pathlib import Path
import uuid

from pymilvus import AnnSearchRequest, WeightedRanker
import pytest

from ragagent.index.embedder import Embedder
from ragagent.index.milvus import connect, load, wait_indexed
from ragagent.index.pipeline import run_index
from ragagent.index.writer import index_document
from ragagent.ingest.chunking.pipeline import run_chunk
from ragagent.ingest.models import Block, CleanDocument
from ragagent.ingest.registry_db import RegistryDB
from tests.index.helpers import config


@pytest.mark.skipif(os.environ.get('L1_MILVUS') != '1', reason='显式启用L1真实Milvus实跑')
def test_http_chunk_write_diskann_hybrid_incremental(tmp_path, embed_url):
    coll = 'l1_m1a_' + uuid.uuid4().hex[:12]
    cfg = config(**{'paths.clean': str(tmp_path / 'clean'),
                    'paths.chunks': str(tmp_path / 'chunks'),
                    'paths.registry_db': str(tmp_path / 'registry.db'),
                    'services.milvus.collection': coll,
                    'chunking.target_tokens': 64, 'chunking.max_tokens': 120,
                    'chunking.min_tokens': 12, 'chunking.overlap_tokens': 12})
    client = connect(cfg)
    with RegistryDB(cfg.paths.registry_db) as registry, Embedder(embed_url, cfg=cfg) as emb:
        try:
            all_chunks = []
            docs = []
            for name, blocks in [
                ('fixture.md', [Block.model_validate(b) for b in json.loads(
                    Path('tests/fixtures/blocks.json').read_text(encoding='utf-8'))]),
                ('synthetic.md', [b for i in range(1152) for b in (
                    Block(type='heading', text=f'部署配置 D{i:05d}', level=1),
                    Block(type='paragraph', text=f'错误码 E{i:05d} 第{i}段：' + '检查连接配置后重试。' * 12))]),
            ]:
                record, _ = registry.register(name, ('a' if name == 'fixture.md' else 'b') * 64)
                doc = CleanDocument(doc_id=record['doc_id'], version=1, source_path=name,
                                    title=Path(name).stem, file_type='md', parser='text',
                                    content_hash=record['content_hash'], blocks=blocks)
                chunks = run_chunk(doc, cfg=cfg.chunking, encode=emb.text)
                docs.append((doc, chunks))
                all_chunks.extend(chunks)
                folder = Path(cfg.paths.clean) / doc.doc_id / 'v1'
                folder.mkdir(parents=True, exist_ok=True)
                (folder / 'clean.json').write_text(doc.model_dump_json(), encoding='utf-8')
                path = Path(cfg.paths.chunks) / doc.doc_id / 'v1.jsonl'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(''.join(c.model_dump_json() + '\n' for c in chunks), encoding='utf-8')
                registry.set_status(doc.doc_id, 'cleaned')
            assert len(all_chunks) >= 1000
            report = run_index(cfg=cfg, client=client, emb=emb, registry=registry, wait=True)
            assert not report['errors'] and report['documents'] == 2
            assert report['chunks'] == len(all_chunks)
            assert all(r['status'] == 'indexed' for r in registry.list_documents())
            for field, d in report['indexes'].items():
                print(f"INDEX {field}: {d['index_type']} {d['indexed_rows']}/{d['total_rows']}")
                assert d['indexed_rows'] == d['total_rows'] > 0
            assert report['indexes']['dense']['index_type'] == 'DISKANN'
            sparse = client.describe_index(coll, 'sparse')
            assert str(sparse['mmap.enabled']).lower() == 'true'
            print('SPARSE_MMAP', sparse['mmap.enabled'])
            for doc, chunks in docs:
                count = client.query(coll, filter=f'doc_id == "{doc.doc_id}"',
                                     output_fields=['count(*)'], consistency_level='Strong')[0]['count(*)']
                assert count == len(chunks)
            target = next(c for c in all_chunks if 'E00042' in c.text)
            dense = emb.text([target.embed_text])[0]
            expr = 'searchable == true'
            result = client.hybrid_search(coll, reqs=[
                AnnSearchRequest(data=[dense], anns_field='dense', limit=50, expr=expr,
                                 param={'metric_type': 'COSINE', 'params': {'search_list': 100}}),
                AnnSearchRequest(data=['E00042'], anns_field='sparse', limit=50, expr=expr,
                                 param={'metric_type': 'BM25'})],
                ranker=WeightedRanker(0.7, 0.3), limit=5,
                output_fields=['chunk_id', 'version', 'text'], consistency_level='Strong')[0]
            assert result[0]['chunk_id'] == target.chunk_id
            print('HYBRID_HITS', [(h['chunk_id'], round(h['distance'], 4)) for h in result])
            # 自向量查询50个样本，不能用随机向量替代自己的embed_text。
            samples = all_chunks[:1] + all_chunks[10:500:10]
            vectors = [emb.image([(c.image_path, c.embed_text)])[0] if c.type == 'image'
                       else emb.text([c.embed_text])[0] for c in samples]
            results = client.search(coll, data=vectors, anns_field='dense', limit=1,
                                    search_params={'metric_type': 'COSINE', 'params': {'search_list': 100}},
                                    consistency_level='Strong')
            assert all(hits[0]['chunk_id'] == c.chunk_id for c, hits in zip(samples, results, strict=True))
            print('SELF_RETRIEVAL', len(samples), '/', len(samples))
            doc, old = docs[0]
            updated = doc.model_copy(update={'version': 2, 'content_hash': 'c' * 64,
                'blocks': [Block(type='paragraph', text='新版本 EUPDATE 配置迁移说明。')]})
            parts = run_chunk(updated, cfg=cfg.chunking, encode=emb.text)
            assert index_document(client, coll, emb, updated, parts, cfg=cfg.index) == len(parts)
            # 再写同一版本是upsert，不能增加逻辑行数。
            index_document(client, coll, emb, updated, parts, cfg=cfg.index)
            client.flush(coll)
            load(client, coll)
            count = client.query(coll, filter=f'doc_id == "{doc.doc_id}"',
                                 output_fields=['count(*)'], consistency_level='Strong')[0]['count(*)']
            old_count = client.query(coll, filter=f'doc_id == "{doc.doc_id}" and version < 2',
                                     output_fields=['count(*)'], consistency_level='Strong')[0]['count(*)']
            assert count == len(parts) and old_count == 0
            print('INCREMENTAL_OK', {'new': count, 'old': old_count})
            print('M1A_OK', {'collection': coll, 'chunks': report['chunks'],
                              'documents': report['documents'], 'model': emb.version})
        finally:
            if client.has_collection(coll):
                client.drop_collection(coll)
            client.close()
