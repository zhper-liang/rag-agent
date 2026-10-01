import asyncio
from pathlib import Path

import httpx
import openai
import pytest
import yaml
from langchain_core.messages import AIMessageChunk

from ragagent.config import ConfigError, load_config
from ragagent.llm.client import close_clients
from ragagent.llm.errors import FirstTokenError, GenerationFailed
from ragagent.llm.messages import to_messages
from ragagent.llm.stream import ThinkFilter, stream_answer, stream_once
from ragagent.prompt.models import ImagePart, Message, PromptPackage, TextPart
from tests.llm.support import ROOT, config, endpoints
from tests.stubs.llm_server import create_app
from tests.stubs.serve import serve


def package(image=False):
    parts = [TextPart(text='合成文档与问题')]
    if image:
        parts.append(ImagePart(media_type='image/jpeg', data_b64='AA==', chunk_id='synthetic'))
    return PromptPackage(scene='rag', system='合成系统提示',
        messages=[Message(role='user', parts=parts)], version='v1', est_input_tokens=10)


def scene(**updates):
    return config().llm.scenes.generate.model_copy(update={
        'primary': 'primary', 'fallback': 'fallback', 'first_token_timeout_s': 1,
        'pre_first_token_budget_s': 3, 'total_timeout_s': 5,
        'retries': 1, 'backoff_s': [0.001], **updates})


def stub_run(base, cfg, *, image=False, eps=None):
    async def run():
        events, sems = [], {}
        try:
            result, info = await stream_answer(package(image), cfg, eps or endpoints(base), sems, events.append)
            assert all(s._value == ep.max_concurrency for name, s in sems.items()
                       for ep in [(eps or endpoints(base))[name]])
            return result, info, events
        finally:
            await close_clients()
    return asyncio.run(run())


def script(base, model, steps):
    httpx.post(base + '/_stub/script', json={'model': model, 'steps': steps}).raise_for_status()


def models(base):
    return [c['model'] for c in httpx.get(base + '/_stub/calls').json()]


def test_normal_stream_exact_order_answer_usage(monkeypatch):
    monkeypatch.setenv('L3_STUB_KEY', 'stub')
    with serve(create_app()) as base:
        script(base, 'primary', [{'chunk_chars': 1}])
        result, info, events = stub_run(base, scene())
        answer = (ROOT / 'tests/fixtures/answers/basic.md').read_text()
        assert result.text == ''.join(e['text'] for e in events) == answer
        assert all(e['type'] == 'token' for e in events)
        assert result.finish_reason == 'stop' and result.usage['output_tokens'] == len(answer)
        assert info['fallback_used'] is False and info['attempts'] == []
        assert isinstance(info['ttft_ms'], int) and info['total_ms'] >= info['ttft_ms']


@pytest.mark.parametrize('status', [429, 500, 503])
def test_retry_before_visible_same_endpoint(monkeypatch, status):
    monkeypatch.setenv('L3_STUB_KEY', 'stub')
    with serve(create_app()) as base:
        script(base, 'primary', [{'status': status}, {}])
        result, info, events = stub_run(base, scene())
        assert models(base) == ['primary', 'primary']
        assert len(info['attempts']) == 1 and info['attempts'][0]['status_code'] == status
        assert info['fallback_used'] is False
        assert result.text == ''.join(e['text'] for e in events)


def test_first_raw_timeout_switches_without_retry(monkeypatch):
    monkeypatch.setenv('L3_STUB_KEY', 'stub')
    with serve(create_app()) as base:
        script(base, 'primary', [{'delay_s': 0.25}])
        result, info, _ = stub_run(base, scene(first_token_timeout_s=0.08))
        assert models(base) == ['primary', 'fallback']
        assert info['fallback_used'] and info['attempts'][0]['error'] == 'TimeoutError'
        assert result.text


@pytest.mark.parametrize('status', [400, 401, 403, 404])
def test_4xx_switches_without_same_endpoint_retry(monkeypatch, status):
    monkeypatch.setenv('L3_STUB_KEY', 'stub')
    with serve(create_app()) as base:
        script(base, 'primary', [{'status': status}])
        _, info, _ = stub_run(base, scene())
        assert models(base) == ['primary', 'fallback']
        assert len(info['attempts']) == 1 and info['fallback_used']


def test_all_endpoints_fail_generation_failed(monkeypatch):
    monkeypatch.setenv('L3_STUB_KEY', 'stub')
    with serve(create_app()) as base:
        script(base, '*', [{'status': 401}, {'status': 400}])
        with pytest.raises(GenerationFailed) as exc:
            stub_run(base, scene())
        assert models(base) == ['primary', 'fallback']
        assert len(exc.value.attempts) == 2
        assert '合成系统提示' not in str(exc.value)


class FakeClient:
    def __init__(self, steps):
        self.steps = steps
        self.closed = False

    async def astream(self, messages):
        try:
            for step in self.steps:
                if isinstance(step, BaseException):
                    raise step
                if isinstance(step, float):
                    await asyncio.sleep(step)
                else:
                    yield step if isinstance(step, AIMessageChunk) else AIMessageChunk(content=step)
        finally:
            self.closed = True


def test_midstream_error_no_retry_duplicate_and_safe_error(monkeypatch):
    calls, client = [], FakeClient(['甲', '乙', RuntimeError('密钥与资料绝不记录')])
    def build(ep, scene):
        calls.append(ep.model)
        return client
    monkeypatch.setattr('ragagent.llm.stream.build_client', build)
    async def run():
        events = []
        res, info = await stream_answer(package(), scene(), endpoints('http://127.0.0.1:18300'), {}, events.append)
        assert res.text == '甲乙' and res.interrupted and res.error == 'RuntimeError'
        assert ''.join(e['text'] for e in events) == '甲乙'
        assert info['attempts'] == []
    asyncio.run(run())
    assert calls == ['primary'] and client.closed


def test_think_filter_at_every_possible_boundary():
    raw = '开头<think>隐藏\n[999]</think>正文<think>更多隐藏</think>尾部'
    for width in range(1, len(raw) + 1):
        think = ThinkFilter()
        visible = ''.join(think.feed(raw[n:n+width]) for n in range(0, len(raw), width)) + think.flush()
        assert visible == '开头正文尾部'
    think = ThinkFilter()
    assert think.feed('<think>不完整隐私') + think.flush() == ''


@pytest.mark.parametrize('steps', [
    ['<thi', 'nk>思考', 0.1, '</th', 'ink>可见'],
    [AIMessageChunk(content='', additional_kwargs={'reasoning_content': '思考'}), 0.1, '可见'],
])
def test_reasoning_not_mistaken_for_first_raw_timeout(steps):
    async def run():
        client, events = FakeClient(steps), []
        res = await stream_once(client, [], 0.03, events.append, ThinkFilter())
        assert res.text == '可见' and res.ttft_ms >= 90
        assert client.closed
    asyncio.run(run())


def test_only_think_normal_end_is_empty_not_timeout():
    async def run():
        with pytest.raises(FirstTokenError) as exc:
            await stream_once(FakeClient(['<think>思考', 0.1, '</think>']), [], 0.03, lambda _: None, ThinkFilter())
        assert type(exc.value.cause) is RuntimeError and exc.value.fatal
    asyncio.run(run())


def test_empty_chunks_do_not_reset_absolute_first_raw_timeout():
    async def run():
        with pytest.raises(FirstTokenError) as exc:
            await stream_once(FakeClient(['', 0.02, '', 0.02, '', 0.02, '迟到']), [], 0.045, lambda _: None)
        assert isinstance(exc.value.cause, TimeoutError)
    asyncio.run(run())


def test_vision_downgrade_and_dropped_count(monkeypatch):
    monkeypatch.setenv('L3_STUB_KEY', 'stub')
    msgs, dropped = to_messages(package(image=True), vision=False)
    assert dropped == 1 and isinstance(msgs[-1].content, str)
    msgs, dropped = to_messages(package(image=True), vision=True)
    assert dropped == 0 and msgs[-1].content[-1]['type'] == 'image_url'
    with serve(create_app()) as base:
        eps = endpoints(base)
        eps['primary'] = eps['primary'].model_copy(update={'vision': True})
        eps['fallback'] = eps['fallback'].model_copy(update={'vision': False})
        script(base, 'primary', [{'status': 401}])
        _, info, _ = stub_run(base, scene(), image=True, eps=eps)
        assert info['images_dropped'] == 1 and info['fallback_used']


@pytest.mark.parametrize('field,value', [('base_url', ''), ('model', ''), ('base_url', 'http://not-local.test/v1')])
def test_startup_invalid_endpoint_refused(tmp_path, field, value):
    raw = yaml.safe_load((ROOT / 'config.yaml').read_text())
    raw['llm']['endpoints'][next(iter(raw['llm']['endpoints']))][field] = value
    p = tmp_path / 'invalid.yaml'
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        load_config(p, env={})


def test_startup_missing_env_refused():
    with pytest.raises(ConfigError, match='环境变量未设置'):
        load_config(env={})


def test_total_timeout_after_visible_is_interruption(monkeypatch):
    monkeypatch.setattr('ragagent.llm.stream.build_client', lambda *_: FakeClient(['正文', 0.1, '迟到']))
    async def run():
        res, info = await stream_answer(package(), scene(total_timeout_s=0.03),
            endpoints('http://127.0.0.1:18300'), {}, lambda _: None)
        assert res.text == '正文' and res.interrupted and info['attempts'] == []
    asyncio.run(run())


def test_queue_budget_cancellation_does_not_release_unacquired_semaphore():
    async def run():
        sem = asyncio.Semaphore(0)
        with pytest.raises(GenerationFailed, match='concurrency queue'):
            await stream_answer(package(), scene(pre_first_token_budget_s=0.01),
                endpoints('http://127.0.0.1:18300'), {'primary': sem}, lambda _: None)
        assert sem._value == 0
    asyncio.run(run())


def test_cancellation_closes_stream_and_releases_semaphore(monkeypatch):
    client = FakeClient(['首字', 2.0, '后续'])
    monkeypatch.setattr('ragagent.llm.stream.build_client', lambda *_: client)
    async def run():
        events, sem = [], asyncio.Semaphore(1)
        task = asyncio.create_task(stream_answer(package(), scene(), endpoints('http://127.0.0.1:18300'),
            {'primary': sem}, events.append))
        while not events:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sem._value == 1 and client.closed
        assert len(events) == 1
    asyncio.run(run())
