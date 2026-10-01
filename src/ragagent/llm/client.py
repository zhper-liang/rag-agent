"""OpenAI 兼容客户端：按接入点/场景缓存，SDK 重试关闭。

异步客户端和信号量不能跨 event loop 重用。每个 loop 独立缓存，
同一进程服务的所有请求在其 loop 内共享接入点并发限额。
"""
from __future__ import annotations

import asyncio
import os
from weakref import WeakKeyDictionary

from langchain_openai import ChatOpenAI

_CLIENTS: WeakKeyDictionary = WeakKeyDictionary()
_SEMS: WeakKeyDictionary = WeakKeyDictionary()


def build_client(ep, scene) -> ChatOpenAI:
    key = (ep.model_dump_json(), scene.model_dump_json(), os.environ[ep.api_key_env])
    loop = asyncio.get_running_loop()
    cache = _CLIENTS.setdefault(loop, {})
    if key not in cache:
        cache[key] = ChatOpenAI(
            base_url=ep.base_url,
            api_key=os.environ[ep.api_key_env],
            model=ep.model,
            temperature=scene.temperature,
            max_tokens=getattr(scene, "max_tokens", None),
            timeout=getattr(scene, "total_timeout_s", None) or scene.timeout_s,
            max_retries=0,
            stream_usage=True,
            extra_body=ep.extra_body or None,
        )
    return cache[key]


def endpoint_semaphore(ep) -> asyncio.Semaphore:
    cache = _SEMS.setdefault(asyncio.get_running_loop(), {})
    key = ep.model_dump_json()
    if key not in cache:
        cache[key] = asyncio.Semaphore(ep.max_concurrency)
    return cache[key]


async def close_clients() -> None:
    """应用关闭时释放本 loop 中的连接池；后续调用可以重新构造。"""
    cache = _CLIENTS.pop(asyncio.get_running_loop(), {})
    for client in cache.values():
        await client.root_async_client.close()
        client.root_client.close()
    _SEMS.pop(asyncio.get_running_loop(), None)
