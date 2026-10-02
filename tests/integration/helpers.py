"""L3集成辅助：真实L2节点、真实HTTP服务桩，仅Milvus使用协议fake。"""
import json
from contextlib import contextmanager
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from ragagent.api.app import create_app

from ragagent.config import load_config
from ragagent.graph.deps import Deps
from ragagent.index.models import ChunkFields
from tests.retrieval.fakes import FakeMilvus, row

KEYS = ['LLM_MAIN_KEY', 'LLM_MAIN_BACKUP_KEY', 'LLM_SMALL_KEY', 'LLM_SMALL_BACKUP_KEY']


def config(tmp_path, llm_url=None, **scene_updates):
    cfg = load_config(env={k: 'test' for k in KEYS})
    paths = cfg.paths.model_copy(update={k: str(tmp_path / v) for k, v in {
        'session_db': 'sessions.db', 'feedback_db': 'feedback.db', 'registry_db': 'registry.db', 'clean': 'clean'}.items()})
    scenes = {name: getattr(cfg.llm.scenes, name).model_copy(update=updates) for name, updates in scene_updates.items()}
    eps = cfg.llm.endpoints
    if llm_url is not None:
        eps = {n: e.model_copy(update={'base_url': llm_url + '/v1', 'model': n}) for n, e in eps.items()}
    return cfg.model_copy(update={'paths': paths,
        'monitor': cfg.monitor.model_copy(update={'langfuse': cfg.monitor.langfuse.model_copy(update={'enabled': False})}),
        'llm': cfg.llm.model_copy(update={'endpoints': eps, 'scenes': cfg.llm.scenes.model_copy(update=scenes)})})


def rows():
    chunks = [row(json.loads(line)) for line in Path('tests/fixtures/chunks.jsonl').read_text().splitlines()]
    for chunk in chunks:
        chunk.update(title='Milvus 部署手册', source_path='手册.pdf', file_type='pdf')
        for key, default in {'page_start': -1, 'page_end': -1, 'image_path': '', 'prev_id': '', 'next_id': ''}.items():
            if chunk[key] is None:
                chunk[key] = default
        # fake候选按文档index协议标量section_path。
        chunk['section_path'] = ' > '.join(chunk['section_path']) if isinstance(chunk['section_path'], list) else chunk['section_path']
    return [ChunkFields.model_validate(chunk).model_dump() for chunk in chunks]


def deps(cfg, embed_url, *, milvus=None, rerank_client=None, **kwargs):
    return Deps(cfg=cfg, count=len, milvus=milvus or FakeMilvus(rows()),
        embed_client=httpx.AsyncClient(base_url=embed_url),
        rerank_client=rerank_client or httpx.AsyncClient(base_url=embed_url),
        **kwargs)


def script(url, model, steps):
    response = httpx.post(url + '/_stub/script', json={'model': model, 'steps': steps})
    response.raise_for_status()


def events(response):
    assert response.status_code == 200
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]


@contextmanager
def api_client(cfg, dep):
    with TestClient(create_app(cfg, deps=dep)) as client:
        try:
            yield client
        finally:
            for obj in (dep.embed_client, dep.rerank_client):
                if obj is not None:
                    client.portal.call(obj.aclose)


def preprocess_script(llm_url, *, intent='rag', question=None):
    script(llm_url, 'small', [{'tool_args': {'intent': intent,
        'standalone_question': question or rows()[2]['text'], 'clarify_question': '您想了解哪一部分？'}}])
