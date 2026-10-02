import asyncio

import httpx
import pytest
from langchain_core.messages import HumanMessage

from ragagent.llm.client import build_client, close_clients
from ragagent.llm.structured import structured_call
from ragagent.preprocess.schema import Preprocessed
from tests.llm.support import endpoints, oneshot
from tests.stubs.llm_server import create_app
from tests.stubs.serve import serve


@pytest.mark.parametrize('failure', [None, {'status': 429}, {'status': 401},
    {'tool_args': '{broken json'}, {'tool_args': {'intent': 'invalid'}}, {'delay_s': 0.2}])
def test_structured_primary_or_single_fallback(monkeypatch, failure):
    monkeypatch.setenv('L3_STUB_KEY', 'stub')
    with serve(create_app()) as base:
        if failure:
            httpx.post(base + '/_stub/script', json={'model': 'primary', 'steps': [failure]}).raise_for_status()
        async def run():
            scene = oneshot(timeout=0.1 if failure and 'delay_s' in failure else 2)
            result, name = await structured_call(scene, endpoints(base), Preprocessed, [HumanMessage('合成问题')])
            await close_clients()
            return result, name
        result, name = asyncio.run(run())
        assert isinstance(result, Preprocessed)
        assert name == ('fallback' if failure else 'primary')
        calls = httpx.get(base + '/_stub/calls').json()
        assert [c['model'] for c in calls] == (['primary', 'fallback'] if failure else ['primary'])
        assert all(c['stream'] is False for c in calls)


def test_structured_both_fail_degrade(monkeypatch):
    monkeypatch.setenv('L3_STUB_KEY', 'stub')
    with serve(create_app()) as base:
        httpx.post(base + '/_stub/script', json={'steps': [{'status': 500}, {'tool_args': 'invalid'}]}).raise_for_status()
        async def run():
            result = await structured_call(oneshot(), endpoints(base), Preprocessed, [HumanMessage('synthetic')])
            await close_clients()
            return result
        assert asyncio.run(run()) == (None, None)
        assert [c['model'] for c in httpx.get(base + '/_stub/calls').json()] == ['primary', 'fallback']


def test_structured_cancellation_propagates(monkeypatch):
    monkeypatch.setenv('L3_STUB_KEY', 'stub')
    with serve(create_app()) as base:
        httpx.post(base + '/_stub/script', json={'model': 'primary', 'steps': [{'delay_s': 2}]}).raise_for_status()
        async def run():
            task = asyncio.create_task(structured_call(oneshot(timeout=3), endpoints(base), Preprocessed, [HumanMessage('synthetic')]))
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await close_clients()
        asyncio.run(run())
        assert all(c['model'] == 'primary' for c in httpx.get(base + '/_stub/calls').json())


def test_client_cache_and_sdk_retry_disabled(monkeypatch):
    monkeypatch.setenv('L3_STUB_KEY', 'stub')
    async def run():
        ep = endpoints('http://127.0.0.1:18300')['primary']
        a = build_client(ep, oneshot())
        assert a is build_client(ep, oneshot())
        assert a.max_retries == 0
        assert a is not build_client(ep, oneshot(timeout=4))
        await close_clients()
    asyncio.run(run())


def test_clients_close_rebuild_and_isolate_event_loops(monkeypatch):
    """显式HTTPX池必须隔离loop；默认ChatOpenAI全局缓存会重用已关闭池。"""
    monkeypatch.setenv('L3_STUB_KEY', 'stub')
    with serve(create_app()) as base:
        clients = []
        async def run():
            scene, eps = oneshot(), endpoints(base)
            first = build_client(eps['primary'], scene)
            assert first is build_client(eps['primary'], scene)
            clients.append(first)
            result, endpoint = await structured_call(scene, eps, Preprocessed, [HumanMessage('合成问题')])
            assert result is not None and endpoint == 'primary'
            await close_clients()
            assert first.http_async_client.is_closed and first.http_client.is_closed
            second = build_client(eps['primary'], scene)
            assert second is not first
            result, endpoint = await structured_call(scene, eps, Preprocessed, [HumanMessage('合成问题')])
            assert result is not None and endpoint == 'primary'
            await close_clients()
        asyncio.run(run())
        asyncio.run(run())
        assert clients[0].http_async_client is not clients[1].http_async_client
        assert [c['model'] for c in httpx.get(base + '/_stub/calls').json()] == ['primary'] * 4


def test_structured_global_metrics_success_and_fallback(monkeypatch):
    from prometheus_client import generate_latest
    from ragagent.monitor.metrics import MonitorMetrics
    monkeypatch.setenv('L3_STUB_KEY', 'stub')
    with serve(create_app()) as base:
        metrics = MonitorMetrics(endpoints=['primary', 'fallback'])
        monkeypatch.setattr('ragagent.llm.observability.METRICS', metrics)
        httpx.post(base + '/_stub/script', json={'model': 'primary', 'steps': [{'status': 400}]}).raise_for_status()
        async def run():
            result, endpoint = await structured_call(oneshot(), endpoints(base), Preprocessed,
                [HumanMessage('合成问题')])
            assert result is not None and endpoint == 'fallback'
            await close_clients()
        asyncio.run(run())
        wire = generate_latest(metrics.registry).decode()
        assert 'rag_llm_calls_total{endpoint="primary",outcome="failed",scene="preprocess"} 1.0' in wire
        assert 'rag_llm_calls_total{endpoint="fallback",outcome="fallback",scene="preprocess"} 1.0' in wire
