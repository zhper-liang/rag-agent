import asyncio
import json

import httpx
import pytest

from eval.calls import CallLimitExceeded, counted_calls


def test_actual_transport_budget_counts_fallback_stream_usage_and_restores(tmp_path):
    original = httpx.AsyncClient.send
    seen = []
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"usage": {"prompt_tokens": 8, "completion_tokens": 2}}\n\n'
        async def aclose(self):
            pass
    async def transport(request):
        seen.append(request.url.path)
        if request.url.path.endswith('embeddings'):
            return httpx.Response(200, json={})
        return httpx.Response(200, stream=Stream())
    path = tmp_path / 'calls.json'
    async def run():
        with counted_calls(2, ledger=path) as counter:
            async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
                for _ in range(2):
                    async with client.stream('POST', 'https://test/v1/chat/completions', json={'model':'m','stream':True}) as response:
                        await response.aread()
                await client.post('https://test/embeddings', json={})
                with pytest.raises(CallLimitExceeded):
                    await client.post('https://test/v1/chat/completions', json={'model':'m'})
            assert counter.summary()['usage']['prompt_tokens'] == 16
    asyncio.run(run())
    assert httpx.AsyncClient.send is original
    report = json.loads(path.read_text())
    assert report['actual_calls'] == 2 and report['blocked_calls'] == 1
    assert len(seen) == 3
