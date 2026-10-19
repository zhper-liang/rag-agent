import asyncio
import json
from types import SimpleNamespace
import time
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ragagent.api.health import create_health_router
from ragagent.ingest.registry_db import RegistryDB
from tests.integration.helpers import config


class Milvus:
    def __init__(self, fail=None, delay=0):
        self.fail, self.delay, self.calls = fail, delay, []

    def describe_alias(self, name, **kwargs):
        self.calls.append(('alias', name))
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError('secret-milvus-exception')

    def query(self, **kwargs):
        self.calls.append(('query', kwargs))
        return []


def setup(tmp_path, *, db=True, code=200, milvus=None):
    cfg = config(tmp_path)
    cfg = cfg.model_copy(update={'api':cfg.api.model_copy(update={'health_timeout_s': .05})})
    if db:
        RegistryDB(cfg.paths.registry_db).close()
    embed = httpx.AsyncClient(base_url='http://embed', transport=httpx.MockTransport(
        lambda request: httpx.Response(code, json={'status':'ok'})))
    app = FastAPI()
    app.state.deps = SimpleNamespace(milvus=milvus or Milvus(), embed_client=embed)
    app.include_router(create_health_router(cfg))
    return app, embed


def test_health_live_and_ready(tmp_path):
    milvus = Milvus()
    app, embed = setup(tmp_path, milvus=milvus)
    with TestClient(app) as client:
        assert client.get('/api/health/live').json() == {'status':'ok'}
        result = client.get('/api/health/ready')
        assert result.status_code == 200
        assert set(result.json()['checks']) == {'milvus','embed','registry'}
        assert milvus.calls[0][0] == 'alias' and milvus.calls[1][0] == 'query'
        client.portal.call(embed.aclose)


@pytest.mark.parametrize('db,code,fail,failed', [(False,200,None,'registry'), (True,503,None,'embed'),
                                                (True,200,True,'milvus')])
def test_health_reports_each_failure_without_raw_errors(tmp_path, db, code, fail, failed):
    app, embed = setup(tmp_path, db=db, code=code, milvus=Milvus(fail=fail))
    with TestClient(app) as client:
        assert client.get('/api/health/live').status_code == 200
        response = client.get('/api/health/ready')
        assert response.status_code == 503
        assert response.json()['checks'][failed]['status'] == 'error'
        assert 'secret-milvus-exception' not in response.text
        client.portal.call(embed.aclose)
    if not db:
        assert not (tmp_path / 'registry.db').exists()


def test_health_timeout_nonblocking_and_deduplicated(tmp_path):
    milvus = Milvus(delay=.25)
    app, embed = setup(tmp_path, milvus=milvus)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://api') as client:
            tasks = [client.get('/api/health/ready') for _ in range(5)]
            start = time.perf_counter()
            responses = await asyncio.gather(*tasks)
            assert time.perf_counter() - start < .2
            assert all(r.status_code == 503 for r in responses)
            assert all(r.json()['checks']['milvus']['reason'] == 'timeout' for r in responses)
            assert sum(call[0] == 'alias' for call in milvus.calls) == 1
            assert (await client.get('/api/health/live')).status_code == 200
        await asyncio.sleep(.3)
        await embed.aclose()
    asyncio.run(run())
