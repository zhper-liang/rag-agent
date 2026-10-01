"""MinerU v4批量上传、恢复轮询与安全原子下载。密钥不进入上传/下载会话。"""
from __future__ import annotations

import asyncio
import inspect
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import time
import zipfile

import httpx
from pydantic import BaseModel, Field

from ragagent.config import load_config
from ragagent.ingest.dedup import sha256

log = logging.getLogger(__name__)
MAX_FILES = 50  # 官网本地上传接口详细章节（顶部概览有200的矛盾）。
MAX_BYTES = 200*1024*1024
MAX_ZIP_BYTES = 2*1024*1024*1024


async def _sleep(seconds):
    """重试/轮询专用边界，单测替换它不会影响共享事件循环中的其他任务。"""
    await asyncio.sleep(seconds)


class ParseJob(BaseModel):
    doc_id: str = Field(pattern=r'^[0-9a-f]{16}$')
    path: Path
    out_dir: Path


class MinerUError(RuntimeError):
    def __init__(self, code, message='MinerU请求失败'):
        self.code = code
        super().__init__(f'{message} (code={code})')


class QuotaExceeded(MinerUError):
    pass


def _manifest(job):
    return job.out_dir.parent / ('.'+job.out_dir.name+'.batch.json')


def _cached(job):
    """只有源码hash与所有产物hash都匹配的完整缓存可复用。"""
    try:
        root = job.out_dir.resolve()
        if job.out_dir.is_symlink():
            return False
        info = json.loads((root/'_complete.json').read_text())
        if info['source_sha256'] != sha256(job.path) or not info['files']:
            return False
        for name, digest in info['files'].items():
            file = root/name
            if file.is_symlink() or not file.resolve().is_relative_to(root) or not file.is_file() or sha256(file) != digest:
                return False
        return any(n.endswith('content_list.json') for n in info['files']) and any(Path(n).name == 'full.md' for n in info['files'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def extract_zip(zip_path: Path, out_dir: Path, *, source_hash: str) -> Path:
    """先完整校验再解压；拒绝ZipSlip、symlink、特殊文件、炸弹和不完整产物。"""
    if out_dir.is_symlink():
        raise ValueError('解析输出不允许symlink')
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.mineru-', dir=out_dir.parent))
    try:
        with zipfile.ZipFile(zip_path) as archive:
            files = archive.infolist()
            if len(files) > 10000 or sum(f.file_size for f in files) > MAX_ZIP_BYTES:
                raise ValueError('解析zip解压容量超限')
            seen = set()
            for item in files:
                name = item.filename
                path = PurePosixPath(name)
                mode = item.external_attr >> 16
                if (not name or '\\' in name or '\x00' in name or path.is_absolute() or '..' in path.parts or
                        re.match(r'^[A-Za-z]:', name) or not (staging/name).resolve().is_relative_to(staging.resolve()) or
                        (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR})):
                    raise ValueError('解析zip包含不安全路径/文件类型')
                normalized = path.as_posix().rstrip('/')
                if normalized in seen:
                    raise ValueError('解析zip包含重复目标')
                seen.add(normalized)
            archive.extractall(staging)
        content = list(staging.rglob('*content_list.json'))
        if len(content) != 1 or not list(staging.rglob('full.md')):
            raise ValueError('MinerU产物不完整：缺少唯一content_list或full.md')
        info = {'source_sha256': source_hash, 'files': {p.relative_to(staging).as_posix(): sha256(p)
                for p in sorted(staging.rglob('*')) if p.is_file()}}
        (staging/'_complete.json').write_text(json.dumps(info), encoding='utf-8')
        if out_dir.exists():
            shutil.rmtree(out_dir)
        staging.replace(out_dir)
        return out_dir
    finally:
        if staging.exists():
            shutil.rmtree(staging)


class MinerUClient:
    def __init__(self, token=None, model_version='vlm', *, cfg=None, transport=None,
                 on_batch=None, resume_batches=None):
        self.cfg = cfg or load_config().ingest.mineru
        self.model_version = model_version
        if model_version not in {'vlm', 'pipeline'}:
            raise ValueError('不支持的MinerU模型')
        token = token if token is not None else os.getenv(self.cfg.token_env)
        if not token:
            raise ValueError(f'缺少环境变量 {self.cfg.token_env}')
        self.api = httpx.AsyncClient(base_url=self.cfg.base_url.rstrip('/')+'/',
            headers={'Authorization': 'Bearer '+token}, timeout=120, transport=transport)
        self.transfer = httpx.AsyncClient(timeout=300, transport=transport, follow_redirects=True)
        self.on_batch = on_batch
        self.resume_batches = resume_batches or {}
        self._quota = asyncio.Event()

    async def _request(self, client, method, url, *, api=False, **kwargs):
        for attempt in range(self.cfg.retries+1):
            try:
                response = await client.request(method, url, **kwargs)
                if response.status_code >= 500 or response.status_code == 429:
                    if attempt < self.cfg.retries:
                        await _sleep(2**attempt)
                        continue
                if not response.is_success:
                    raise MinerUError(response.status_code, 'MinerU HTTP失败')
                if not api:
                    return response
                payload = response.json()
                code = payload.get('code')
                if code != 0:
                    if code in {-60018, -60019}:
                        self._quota.set()
                        raise QuotaExceeded(code, 'MinerU每日额度不足，停止后续提交')
                    raise MinerUError(code)
                if not isinstance(payload.get('data'), dict):
                    raise MinerUError('invalid-data')
                return payload['data']
            except httpx.TransportError as exc:
                if attempt == self.cfg.retries:
                    # 不保存带签名URL的原始httpx异常文本。
                    raise MinerUError('network', type(exc).__name__) from None
                await _sleep(2**attempt)
        raise AssertionError('unreachable')

    async def parse_batch(self, jobs, poll_interval=5, timeout=1800) -> dict[str, Path | Exception]:
        jobs = [ParseJob.model_validate(job) for job in jobs]
        if len({j.doc_id for j in jobs}) != len(jobs):
            raise ValueError('ParseJob.doc_id不可重复')
        if poll_interval < 0 or timeout <= 0:
            raise ValueError('poll_interval/timeout无效')
        results, pending, resume = {}, [], {}
        for job in jobs:
            try:
                if _cached(job):
                    results[job.doc_id] = job.out_dir
                elif job.path.stat().st_size > MAX_BYTES:
                    results[job.doc_id] = MinerUError(-60005, '单文件超过200MB，应先拆分')
                elif batch_id := self.resume_batches.get(job.doc_id):
                    resume.setdefault(batch_id, []).append(job)
                else:
                    pending.append(job)
            except OSError as exc:
                results[job.doc_id] = exc
        # 先恢复已有批次，不能因为新的额度耗尽遗漏已有结果。
        for batch_id, group in resume.items():
            results.update(await self._poll(group, batch_id, poll_interval, timeout))
        size = min(self.cfg.batch_size, MAX_FILES)
        semaphore = asyncio.Semaphore(self.cfg.concurrency)

        async def process(group):
            async with semaphore:
                if self._quota.is_set():
                    return {j.doc_id: QuotaExceeded(-60019, '未提交，等待额度恢复') for j in group}
                try:
                    return await self._submit(group, poll_interval, timeout)
                except Exception as exc:
                    return {j.doc_id: exc for j in group}
        groups = [pending[i:i+size] for i in range(0, len(pending), size)]
        for group_result in await asyncio.gather(*(process(g) for g in groups)):
            results.update(group_result)
        return results

    async def _submit(self, jobs, interval, timeout):
        data = await self._request(self.api, 'POST', 'file-urls/batch', api=True, json={
            'files': [{'name': j.path.name, 'data_id': j.doc_id, 'is_ocr': self.cfg.is_ocr} for j in jobs],
            'model_version': self.model_version, 'enable_formula': self.cfg.enable_formula,
            'enable_table': self.cfg.enable_table, 'language': self.cfg.language})
        batch_id, urls = data.get('batch_id'), data.get('file_urls')
        if not isinstance(batch_id, str) or not batch_id or not isinstance(urls, list) or len(urls) != len(jobs):
            raise MinerUError('invalid-upload-response')
        for job, url in zip(jobs, urls, strict=True):
            self._transfer_url(url)
            job.out_dir.parent.mkdir(parents=True, exist_ok=True)
            manifest = _manifest(job)
            if manifest.is_symlink():
                raise ValueError('批次恢复文件不允许symlink')
            # presigned URL仅本地0600存放，恢复waiting-file时可继续上传，不输出日志。
            fd = os.open(manifest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, 'w') as f:
                json.dump({'batch_id': batch_id, 'source_sha256': sha256(job.path), 'upload_url': url}, f)
        if self.on_batch:
            result = self.on_batch(batch_id, jobs)
            if inspect.isawaitable(result):
                await result
        failed = {}
        for job, url in zip(jobs, urls, strict=True):
            try:
                await self._upload(job, url)
            except Exception as exc:
                failed[job.doc_id] = exc
        survivors = [j for j in jobs if j.doc_id not in failed]
        if survivors:
            failed.update(await self._poll(survivors, batch_id, interval, timeout))
        return failed

    @staticmethod
    def _transfer_url(url):
        if not isinstance(url, str) or not url.startswith('https://'):
            raise ValueError('上传/下载地址必须HTTPS')

    async def _upload(self, job, url):
        self._transfer_url(url)
        # 文件通常<200MB，重试时必须重新读取而不是重用已消费的流。
        content = job.path.read_bytes()
        manifest = json.loads(_manifest(job).read_text())
        if hashlib.sha256(content).hexdigest() != manifest['source_sha256']:
            raise ValueError('源文件在上传前已变化，需要新版本重新提交')
        await self._request(self.transfer, 'PUT', url, content=content)

    async def _poll(self, jobs, batch_id, interval, timeout):
        results, by_id, reuploaded = {}, {j.doc_id: j for j in jobs}, set()
        deadline = time.monotonic()+timeout
        while len(results) < len(jobs) and time.monotonic() < deadline:
            try:
                data = await self._request(self.api, 'GET', f'extract-results/batch/{batch_id}', api=True)
                items = data.get('extract_result')
                if not isinstance(items, list):
                    raise MinerUError('invalid-results')
                for item in items:
                    doc_id = item.get('data_id')
                    if doc_id not in by_id or doc_id in results:
                        continue
                    job = by_id[doc_id]
                    state = item.get('state')
                    if state == 'done':
                        try:
                            results[doc_id] = await self._fetch(item.get('full_zip_url'), job)
                            _manifest(job).unlink(missing_ok=True)
                        except Exception as exc:
                            results[doc_id] = exc
                    elif state == 'failed':
                        results[doc_id] = MinerUError(item.get('err_code', -60007), 'MinerU单文件解析失败')
                    elif state == 'waiting-file' and doc_id not in reuploaded:
                        try:
                            info = json.loads(_manifest(job).read_text())
                            if info['batch_id'] != batch_id or info['source_sha256'] != sha256(job.path):
                                raise ValueError('恢复批次与输入不匹配')
                            await self._upload(job, info['upload_url'])
                            reuploaded.add(doc_id)
                        except Exception:
                            results[doc_id] = MinerUError('resume-upload', '无法恢复上传，应重新提交')
                    elif state not in {'waiting-file', 'pending', 'running', 'converting'}:
                        results[doc_id] = MinerUError('unknown-state')
            except Exception as exc:
                return {j.doc_id: results.get(j.doc_id, exc) for j in jobs}
            if len(results) < len(jobs):
                await _sleep(min(interval, max(0, deadline-time.monotonic())))
        return {j.doc_id: results.get(j.doc_id, TimeoutError('MinerU批次轮询超时')) for j in jobs}

    async def _fetch(self, url, job):
        self._transfer_url(url)
        source_hash = sha256(job.path)
        if _manifest(job).exists():
            submitted = json.loads(_manifest(job).read_text())
            if submitted['source_sha256'] != source_hash:
                raise ValueError('源文件在云端解析期间已变化，需要新版本重新提交')
        job.out_dir.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.download-', suffix='.zip', dir=job.out_dir.parent)
        try:
            # response流有大小上限；下载网络/5xx也复用指数退避策略。
            for attempt in range(self.cfg.retries+1):
                try:
                    with os.fdopen(os.dup(fd), 'wb') as f:
                        f.seek(0)
                        f.truncate()
                        size = 0
                        async with self.transfer.stream('GET', url) as response:
                            if response.status_code >= 500 or response.status_code == 429:
                                if attempt < self.cfg.retries:
                                    await _sleep(2**attempt)
                                    continue
                            if not response.is_success:
                                raise MinerUError(response.status_code, 'MinerU下载失败')
                            async for chunk in response.aiter_bytes():
                                size += len(chunk)
                                if size > MAX_ZIP_BYTES:
                                    raise ValueError('解析zip下载容量超限')
                                f.write(chunk)
                    if sha256(job.path) != source_hash:
                        raise ValueError('源文件在结果下载期间已变化')
                    return extract_zip(Path(name), job.out_dir, source_hash=source_hash)
                except httpx.TransportError as exc:
                    if attempt == self.cfg.retries:
                        raise MinerUError('network', type(exc).__name__) from None
                    await _sleep(2**attempt)
            raise AssertionError('unreachable')
        finally:
            os.close(fd)
            Path(name).unlink(missing_ok=True)

    async def aclose(self):
        await self.api.aclose()
        await self.transfer.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()
