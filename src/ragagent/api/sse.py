"""仅允许约定五种事件；JSON里的换行不能注入新的SSE frame。"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress

EVENT_TYPES = frozenset({'progress', 'meta', 'token', 'final', 'error'})
log = logging.getLogger(__name__)


def encode_event(event: dict) -> bytes:
    if event.get('type') not in EVENT_TYPES:
        raise ValueError('unknown stream event')
    return ('data: ' + json.dumps(event, ensure_ascii=False, separators=(',', ':')) + '\n\n').encode('utf-8')


async def graph_events(graph, state, config, context, *, request=None, on_update=None):
    """监督图任务：断开即取消，生成器关闭也取消，不留下后台LLM请求。"""
    queue = asyncio.Queue(maxsize=32)
    end = object()

    async def produce():
        try:
            async for mode, data in graph.astream(state, config=config, context=context,
                                                 stream_mode=['custom', 'updates']):
                if mode == 'custom':
                    await queue.put(data)
                elif mode == 'updates' and on_update is not None:
                    on_update(data)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning('graph stream failed_type=%s', type(exc).__name__)
            await queue.put({'type': 'error', 'message': '服务暂时不可用，请稍后再试'})
        finally:
            # 取消路径不能等待满队列；外层已停止读取。
            if not asyncio.current_task().cancelling():
                await queue.put(end)

    producer = asyncio.create_task(produce())

    async def watch_disconnect():
        while not producer.done():
            if await request.is_disconnected():
                producer.cancel()
                return
            await asyncio.sleep(.05)

    watcher = asyncio.create_task(watch_disconnect()) if request is not None else None
    try:
        while True:
            item_task = asyncio.create_task(queue.get())
            # 同时监听producer终结：断开后没有end，仍须立即退出。
            try:
                done, _ = await asyncio.wait([item_task, producer], return_when=asyncio.FIRST_COMPLETED)
                if producer in done and producer.cancelled():
                    break
                # 正常producer终结一定已排入end；继续等get，不能因瞬时empty丢final。
                item = await item_task
            finally:
                if not item_task.done():
                    item_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await item_task
            if item is end:
                break
            yield item
    finally:
        producer.cancel()
        if watcher is not None:
            watcher.cancel()
        with suppress(asyncio.CancelledError):
            await producer
        if watcher is not None:
            with suppress(asyncio.CancelledError):
                await watcher
