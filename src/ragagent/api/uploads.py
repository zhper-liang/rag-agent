"""D-021 上传接口；先访问检查，再限流式 multipart，再校验冲突。"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import shutil
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from ragagent.ingest.upload import sanitize_names


def check_access(request, cfg):
    if request.headers.get(cfg.api.upload.client_header) != 'web':
        raise HTTPException(403, '缺少网页上传请求头')
    host = request.headers.get('host', '')
    try:
        parsed = urlsplit('//' + host)
        port = parsed.port
        valid_host = (parsed.hostname in ('127.0.0.1', 'localhost') and parsed.netloc == host
                      and parsed.username is None and parsed.password is None
                      and not parsed.path and not parsed.query and not parsed.fragment)
    except ValueError:
        valid_host = False
    if not valid_host:
        raise HTTPException(403, '不允许的 Host')
    origin = request.headers.get('origin')
    if origin is not None:
        try:
            other = urlsplit(origin)
            valid_origin = (other.scheme == request.url.scheme and other.hostname == parsed.hostname
                and (other.port or (443 if other.scheme == 'https' else 80)) ==
                    (port or (443 if request.url.scheme == 'https' else 80))
                and other.username is None and other.password is None
                and not other.path and not other.query and not other.fragment)
        except ValueError:
            valid_origin = False
        if not valid_origin:
            raise HTTPException(403, '不允许跨站上传')


class BoundedMultipart(MultiPartParser):
    def __init__(self, request, cfg):
        super().__init__(request.headers, request.stream(), max_files=cfg.api.upload.max_files,
                         max_fields=1, max_part_size=16)
        self.file_limit = cfg.api.upload.max_file_mb * 1024 * 1024
        self.bytes_in_part = 0

    def on_part_begin(self):
        super().on_part_begin()
        self.bytes_in_part = 0

    def on_part_data(self, data, start, end):
        self.bytes_in_part += end - start
        if self._current_part.file is not None and self.bytes_in_part > self.file_limit:
            raise MultiPartException('文件超过大小上限')
        super().on_part_data(data, start, end)


async def digest_file(file):
    digest = hashlib.sha256()
    while part := await file.read(64 * 1024):
        digest.update(part)
    await file.seek(0)
    return digest.hexdigest()


def create_upload_router(cfg):
    router = APIRouter()

    @router.post('/api/uploads')
    async def upload(request: Request):
        check_access(request, cfg)
        if request.headers.get('content-type', '').split(';')[0].strip().lower() != 'multipart/form-data':
            raise HTTPException(400, '需要 multipart/form-data')
        try:
            form = await BoundedMultipart(request, cfg).parse()
        except (MultiPartException, ValueError):
            raise HTTPException(400, '上传格式、文件数或大小超限') from None
        try:
            files = form.getlist('files')
            if (not files or any(not isinstance(f, UploadFile) for f in files)
                    or any(k not in ('files', 'overwrite') for k in form)
                    or len(form.getlist('overwrite')) > 1):
                raise HTTPException(400, '无效的上传字段')
            overwrite = form.get('overwrite', request.query_params.get('overwrite', 'false'))
            if overwrite not in ('true', 'false'):
                raise HTTPException(400, 'overwrite 必须为 true 或 false')
            try:
                names = sanitize_names([f.filename or '' for f in files], cfg)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from None
            hashes = [await digest_file(f) for f in files]
            manager = request.app.state.uploads
            async with manager.submissions:
                pairs = list(zip(names, hashes))
                conflicts = await asyncio.to_thread(manager.store.conflicts, pairs)
                if conflicts and overwrite != 'true':
                    return JSONResponse({'conflicts': conflicts}, status_code=409)
                job_id = uuid4().hex
                folder = manager.store.staging / job_id
                try:
                    await asyncio.to_thread(folder.mkdir, parents=True, exist_ok=False)
                    for seq, file in enumerate(files):
                        # Form 解析已限流式写临时文件；持久化暂存区仅在 409 检查通过后建立。
                        await asyncio.to_thread(_stage, file.file, folder / str(seq))
                    await asyncio.to_thread(manager.store.create, job_id, pairs)
                except BaseException:
                    await asyncio.to_thread(shutil.rmtree, folder, True)
                    raise
                manager.wake.set()
                from ragagent.ingest.registry_db import make_doc_id
                return JSONResponse({'job_id': job_id, 'files': [
                    {'name': n, 'doc_id': make_doc_id('@uploads/' + n)} for n in names]}, status_code=202)

        finally:
            await form.close()

    @router.get('/api/uploads')
    async def list_jobs(request: Request):
        check_access(request, cfg)
        return {'jobs': await asyncio.to_thread(request.app.state.uploads.store.list)}

    @router.get('/api/uploads/{job_id}')
    async def get_job(job_id: str, request: Request):
        check_access(request, cfg)
        job = await asyncio.to_thread(request.app.state.uploads.store.get, job_id)
        if job is None:
            raise HTTPException(404, '上传任务不存在')
        return job

    @router.post('/api/uploads/{job_id}/retry')
    async def retry_job(job_id: str, request: Request):
        check_access(request, cfg)
        manager = request.app.state.uploads
        async with manager.submissions:
            job = await asyncio.to_thread(manager.store.get, job_id)
            if job is None:
                raise HTTPException(404, '上传任务不存在')
            if not await asyncio.to_thread(manager.store.retry, job_id):
                raise HTTPException(409, '只允许重试失败或中断的任务')
            manager.wake.set()
            return JSONResponse(await asyncio.to_thread(manager.store.get, job_id), status_code=202)

    return router


def _stage(source, target: Path):
    temp = target.with_suffix('.tmp')
    with temp.open('xb') as out:
        shutil.copyfileobj(source, out, 64 * 1024)
    temp.replace(target)
