"""离线 CLI 的实际 HTTP 请求预算；主备、stream、RAGAS 共用同一上限。

只记录模型、状态、usage 和异常类别，不记录密钥或文本。ContextVar 隔离
当前 CLI 的任务；包装器只在持有上下文时生效，退出恢复 httpx.send。
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path

import httpx


class CallLimitExceeded(RuntimeError):
    pass


_ACTIVE = ContextVar('eval_http_calls', default=None)


class Calls:
    def __init__(self, limit, ledger=None):
        if type(limit) is not int or limit < 1:
            raise ValueError('max-calls must be a positive integer')
        self.limit, self.ledger = limit, Path(ledger) if ledger else None
        self.requests, self.blocked = [], 0

    def summary(self):
        usage = {}
        for row in self.requests:
            for key, value in row.get('usage', {}).items():
                if type(value) is int:
                    usage[key] = usage.get(key, 0) + value
        return {'actual_calls': len(self.requests), 'max_calls': self.limit,
                'blocked_calls': self.blocked, 'usage': usage, 'requests': self.requests}

    def save(self):
        if self.ledger:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.ledger.with_suffix(self.ledger.suffix + '.tmp')
            temporary.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2) + '\n')
            temporary.replace(self.ledger)

    def begin(self, request):
        if len(self.requests) >= self.limit:
            self.blocked += 1
            self.save()
            raise CallLimitExceeded('actual HTTP call budget exhausted')
        body = json.loads(request.content)
        row = {'number': len(self.requests) + 1, 'model': body.get('model'),
               'stream': bool(body.get('stream')), 'status': None}
        self.requests.append(row)
        self.save()
        return row

    def usage(self, row, payload):
        if isinstance(payload, dict) and isinstance(payload.get('usage'), dict):
            row['usage'] = {k: v for k, v in payload['usage'].items() if type(v) is int}
            self.save()


class UsageStream(httpx.AsyncByteStream):
    def __init__(self, stream, calls, row):
        self.stream, self.calls, self.row = stream, calls, row

    async def __aiter__(self):
        pending = b''
        try:
            async for chunk in self.stream:
                pending += chunk
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    if line.startswith(b'data:'):
                        try:
                            self.calls.usage(self.row, json.loads(line[5:].strip()))
                        except (ValueError, TypeError):
                            pass
                if len(pending) > 1_000_000:
                    pending = b''
                yield chunk
        except BaseException as exc:
            self.row['error_type'] = type(exc).__name__
            self.calls.save()
            raise

    async def aclose(self):
        await self.stream.aclose()


def current_summary():
    calls = _ACTIVE.get()
    return calls.summary() if calls else {}


@contextmanager
def counted_calls(limit, *, ledger=None):
    """覆盖整段 async CLI 生命周期，包括 SDK 隐式重试的物理请求。"""
    calls = Calls(limit, ledger)
    token = _ACTIVE.set(calls)
    original = httpx.AsyncClient.send

    async def send(client, request, **kwargs):
        tracker = _ACTIVE.get()
        if tracker is None or request.method != 'POST' or not request.url.path.endswith(('/chat/completions', '/responses')):
            return await original(client, request, **kwargs)
        row = tracker.begin(request)
        try:
            response = await original(client, request, **kwargs)
            row['status'] = response.status_code
            if kwargs.get('stream'):
                response.stream = UsageStream(response.stream, tracker, row)
            else:
                try:
                    tracker.usage(row, response.json())
                except ValueError:
                    pass
            tracker.save()
            return response
        except BaseException as exc:
            row['error_type'] = type(exc).__name__
            tracker.save()
            raise

    httpx.AsyncClient.send = send
    try:
        yield calls
    finally:
        httpx.AsyncClient.send = original
        _ACTIVE.reset(token)
        calls.save()
