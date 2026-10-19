"""进程存活与依赖就绪检查；不请求 LLM，不泄露异常正文。"""
import asyncio
import sqlite3
from pathlib import Path

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


def create_health_router(cfg):
    router = APIRouter()
    pending = {}
    timeout = cfg.api.health_timeout_s

    def registry_probe():
        uri = Path(cfg.paths.registry_db).resolve().as_uri() + '?mode=ro'
        conn = sqlite3.connect(uri, uri=True, timeout=timeout)
        try:
            conn.execute('SELECT doc_id FROM documents LIMIT 1').fetchall()
        finally:
            conn.close()

    def milvus_probe(client):
        name = cfg.services.milvus.collection
        client.describe_alias(name, timeout=timeout)
        client.query(collection_name=name, filter='', output_fields=['chunk_id'], limit=1, timeout=timeout)

    async def embed_probe(client):
        if client is None:
            url = f'http://{cfg.services.embed.host}:{cfg.services.embed.port}'
            async with httpx.AsyncClient(base_url=url, timeout=timeout) as owned:
                response = await owned.get('/health')
        else:
            response = await client.get('/health', timeout=timeout)
        response.raise_for_status()
        body = response.json()
        if body.get('status') not in (None, 'ok', 'ready'):
            raise RuntimeError('embed not ready')

    async def checked(name, probe):
        # 超时线程无法强制终止；复用尚未完成的任务，避免探活风暴堆积线程。
        task = pending.get(name)
        if task is None or task.done():
            task = asyncio.create_task(probe())
            pending[name] = task
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            return {'status': 'ok'}
        except TimeoutError:
            return {'status': 'error', 'reason': 'timeout'}
        except Exception as exc:
            return {'status': 'error', 'reason': type(exc).__name__}

    @router.get('/api/health/live')
    async def live():
        return {'status': 'ok'}

    @router.get('/api/health/ready')
    async def ready(request: Request):
        deps = request.app.state.deps
        names = ('milvus', 'embed', 'registry')
        results = await asyncio.gather(
            checked('milvus', lambda: asyncio.to_thread(milvus_probe, deps.milvus)),
            checked('embed', lambda: embed_probe(deps.embed_client)),
            checked('registry', lambda: asyncio.to_thread(registry_probe)),
        )
        checks = dict(zip(names, results))
        ok = all(item['status'] == 'ok' for item in results)
        return JSONResponse({'status': 'ok' if ok else 'not_ready', 'checks': checks},
                            status_code=200 if ok else 503)

    return router
