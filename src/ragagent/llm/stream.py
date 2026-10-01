"""首个可见 token 之前重试，之后中断；绝不重复推出正文。"""
from __future__ import annotations

import asyncio
from contextlib import suppress

from pydantic import BaseModel

from ragagent.llm.client import build_client, endpoint_semaphore
from ragagent.llm.errors import FirstTokenError, GenerationFailed, error_meta, is_retryable
from ragagent.llm.messages import to_messages


class StreamResult(BaseModel):
    text: str
    finish_reason: str | None
    usage: dict | None
    ttft_ms: int | None
    interrupted: bool = False
    error: str | None = None


class ThinkFilter:
    OPEN, CLOSE = '<think>', '</think>'

    def __init__(self):
        self.buf, self.inside = '', False

    def feed(self, text: str) -> str:
        self.buf += text
        out = []
        while True:
            tag = self.CLOSE if self.inside else self.OPEN
            index = self.buf.find(tag)
            if index >= 0:
                if not self.inside:
                    out.append(self.buf[:index])
                self.buf, self.inside = self.buf[index + len(tag):], not self.inside
                continue
            keep = next((n for n in range(min(len(tag) - 1, len(self.buf)), 0, -1)
                         if tag.startswith(self.buf[-n:])), 0)
            ready, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
            if not self.inside:
                out.append(ready)
            return ''.join(out)

    def flush(self) -> str:
        tail, self.buf = ('' if self.inside else self.buf), ''
        return tail


def _text(chunk) -> str:
    if isinstance(chunk.content, str):
        return chunk.content
    return ''.join(p.get('text', '') for p in chunk.content
                   if isinstance(p, dict) and p.get('type') == 'text')


async def stream_once(client, messages, first_token_timeout: float, emit, think=None,
                      *, deadline=None, pre_first_deadline=None) -> StreamResult:
    loop = asyncio.get_running_loop()
    start = loop.time()
    raw_deadline = start + first_token_timeout
    iterator = client.astream(messages).__aiter__()
    parts, usage, finish, ttft = [], None, None, None
    raw_started = False

    def push(text):
        nonlocal ttft
        if ttft is None:
            ttft = int((loop.time() - start) * 1000)
        parts.append(text)
        emit({'type': 'token', 'text': text})

    try:
        while True:
            limits = [n for n in (deadline, pre_first_deadline if not parts else None,
                                  raw_deadline if not raw_started else None) if n is not None]
            try:
                if limits:
                    left = min(limits) - loop.time()
                    if left <= 0:
                        raise TimeoutError('stream deadline reached')
                    chunk = await asyncio.wait_for(iterator.__anext__(), left)
                else:
                    chunk = await iterator.__anext__()
            except StopAsyncIteration:
                break
            usage = getattr(chunk, 'usage_metadata', None) or usage
            finish = chunk.response_metadata.get('finish_reason') or finish
            raw = _text(chunk)
            reasoning = getattr(chunk, 'additional_kwargs', {}).get('reasoning_content')
            if raw or reasoning:
                raw_started = True
            if raw:
                text = think.feed(raw) if think else raw
                if text:
                    push(text)
        if think:
            tail = think.flush()
            if tail:
                push(tail)
    except Exception as exc:
        if not parts:
            raise FirstTokenError(exc, fatal=not is_retryable(exc)) from exc
        return StreamResult(text=''.join(parts), finish_reason=finish, usage=usage,
                            ttft_ms=ttft, interrupted=True, error=type(exc).__name__)
    finally:
        if hasattr(iterator, 'aclose'):
            with suppress(Exception):
                await iterator.aclose()
    if not parts:
        raise FirstTokenError(RuntimeError('empty response'), fatal=True)
    return StreamResult(text=''.join(parts), finish_reason=finish, usage=usage, ttft_ms=ttft)


async def stream_answer(pkg, scene, eps, sems, emit) -> tuple[StreamResult, dict]:
    loop = asyncio.get_running_loop()
    start = loop.time()
    pre_deadline = start + scene.pre_first_token_budget_s
    total_deadline = start + scene.total_timeout_s
    tried, overall_ttft = [], None

    def write(event):
        nonlocal overall_ttft
        if event['type'] == 'token' and overall_ttft is None:
            overall_ttft = int((loop.time() - start) * 1000)
        emit(event)

    for name in (scene.primary, scene.fallback):
        ep = eps[name]
        messages, dropped = to_messages(pkg, ep.vision)
        sem = sems.setdefault(name, endpoint_semaphore(ep))
        for attempt in range(scene.retries + 1):
            left = min(pre_deadline, total_deadline) - loop.time()
            if left <= 0:
                raise GenerationFailed('pre-first-token budget exhausted', tried)
            try:
                await asyncio.wait_for(sem.acquire(), left)
            except TimeoutError as exc:
                raise GenerationFailed('concurrency queue timed out', tried) from exc
            try:
                try:
                    res = await stream_once(build_client(ep, scene), messages,
                        scene.first_token_timeout_s, write,
                        ThinkFilter() if ep.strip_think else None,
                        deadline=total_deadline, pre_first_deadline=pre_deadline)
                except FirstTokenError:
                    raise
                except Exception as exc:
                    raise FirstTokenError(exc, fatal=not is_retryable(exc)) from exc
                return res, {'endpoint': name, 'model': ep.model, 'attempts': tried,
                    'fallback_used': name != scene.primary, 'images_dropped': dropped,
                    'ttft_ms': overall_ttft, 'total_ms': int((loop.time() - start) * 1000)}
            except FirstTokenError as exc:
                tried.append({'endpoint': name, 'attempt': attempt, **error_meta(exc.cause)})
                if exc.fatal or attempt == scene.retries:
                    break
            finally:
                sem.release()
            delay = scene.backoff_s[min(attempt, len(scene.backoff_s) - 1)] if scene.backoff_s else 0
            left = min(pre_deadline, total_deadline) - loop.time()
            if left <= delay:
                raise GenerationFailed('retry backoff would exceed budget', tried)
            await asyncio.sleep(delay)
    raise GenerationFailed(attempts=tried)
