"""网页上传的命名边界、持久化串行任务与可恢复入库（D-021）。"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import re
import shutil
import unicodedata
from uuid import uuid4

from ragagent.ingest.lock import IngestBusy, ingest_lock
from ragagent.ingest.models import CleanDocument
from ragagent.ingest.registry_db import RegistryDB, make_doc_id

log = logging.getLogger(__name__)
ACTIVE = ('queued', 'cleaning', 'chunking', 'indexing')
TERMINAL = ('done', 'failed', 'interrupted')


def now():
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(filename: str, cfg) -> str:
    """只保留 basename；主名截断不吞后缀；返回 NFC 且后缀小写。"""
    name = unicodedata.normalize('NFC', filename.replace('\\', '/').rsplit('/', 1)[-1])
    name = ''.join(c for c in name if not unicodedata.category(c).startswith('C'))
    name = re.sub(r'[/\\:*?"<>|]', '', name)
    name = re.sub(r'\s+', ' ', name).strip(' .')
    suffix = Path(name).suffix.lower()
    allowed = set(cfg.ingest.parse_routes.text) | set(cfg.ingest.parse_routes.mineru)
    if not name or not suffix or suffix[1:] not in allowed:
        raise ValueError('文件名为空或文件格式不支持')
    stem = name[:-len(suffix)].strip(' .')[:cfg.api.upload.max_name_chars].rstrip(' .')
    if not stem:
        raise ValueError('文件主名不能为空')
    return stem + suffix


def sanitize_names(filenames, cfg):
    names = [sanitize_filename(n, cfg) for n in filenames]
    if len(names) != len(set(names)):
        raise ValueError('清洗后文件名重复，请改名后重试')
    return names


class UploadStore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.staging = Path(cfg.paths.parsed) / '.uploads'

    def recover(self):
        with RegistryDB(self.cfg.paths.registry_db) as db, db.transaction():
            db.conn.execute("UPDATE upload_jobs SET status='interrupted',updated_at=? "
                            "WHERE status IN ('queued','cleaning','chunking','indexing')", (now(),))
        self.prune()

    def get(self, job_id):
        with RegistryDB(self.cfg.paths.registry_db) as db:
            row = db.conn.execute('SELECT * FROM upload_jobs WHERE job_id=?', (job_id,)).fetchone()
            if row is None:
                return None
            files = db.conn.execute('SELECT name,doc_id,result,reason FROM upload_files '
                                    'WHERE job_id=? ORDER BY seq', (job_id,)).fetchall()
            return {**dict(row), 'files': [dict(f) for f in files]}

    def list(self):
        with RegistryDB(self.cfg.paths.registry_db) as db:
            ids = [r[0] for r in db.conn.execute('SELECT job_id FROM upload_jobs ORDER BY created_at DESC,job_id DESC')]
        return [self.get(j) for j in ids]

    def files(self, job_id):
        with RegistryDB(self.cfg.paths.registry_db) as db:
            return [dict(r) for r in db.conn.execute('SELECT * FROM upload_files WHERE job_id=? ORDER BY seq', (job_id,))]

    def conflicts(self, files):
        conflicts = []
        with RegistryDB(self.cfg.paths.registry_db) as db:
            for name, digest in files:
                doc_id = make_doc_id('@uploads/' + name)
                record = db.get(doc_id)
                # 任务未执行时 documents 还没有对应版本，也必须挡住排队中的重名覆盖。
                queued = db.conn.execute('''SELECT f.content_hash,f.previous_version FROM upload_files f
                    JOIN upload_jobs j USING(job_id) WHERE f.doc_id=?
                    AND j.status IN ('queued','cleaning','chunking','indexing')
                    ORDER BY j.created_at DESC LIMIT 1''', (doc_id,)).fetchone()
                current_hash = queued['content_hash'] if queued else (record or {}).get('content_hash')
                if current_hash is not None and current_hash != digest:
                    conflicts.append(dict(name=name, doc_id=doc_id,
                        title=(record or {}).get('title', Path(name).stem),
                        version=record['version'] if record else (queued['previous_version'] or 0) + 1))
        return conflicts

    def create(self, job_id, files):
        stamp = now()
        with RegistryDB(self.cfg.paths.registry_db) as db, db.transaction():
            db.conn.execute('INSERT INTO upload_jobs VALUES (?,?,?,?)', (job_id, 'queued', stamp, stamp))
            for seq, (name, digest) in enumerate(files):
                doc_id = make_doc_id('@uploads/' + name)
                record = db.get(doc_id)
                db.conn.execute('INSERT INTO upload_files VALUES (?,?,?,?,?,?,?,?)',
                    (job_id, seq, name, doc_id, digest, record['version'] if record else None, None, None))
        self.prune()

    def state(self, job_id, status):
        with RegistryDB(self.cfg.paths.registry_db) as db, db.transaction():
            db.conn.execute('UPDATE upload_jobs SET status=?,updated_at=? WHERE job_id=?', (status, now(), job_id))

    def result(self, job_id, seq, result, reason=None):
        with RegistryDB(self.cfg.paths.registry_db) as db, db.transaction():
            db.conn.execute('UPDATE upload_files SET result=?,reason=? WHERE job_id=? AND seq=?',
                            (result, reason, job_id, seq))
            db.conn.execute('UPDATE upload_jobs SET updated_at=? WHERE job_id=?', (now(), job_id))

    def retry(self, job_id):
        with RegistryDB(self.cfg.paths.registry_db) as db, db.transaction():
            changed = db.conn.execute("UPDATE upload_jobs SET status='queued',updated_at=? "
                "WHERE job_id=? AND status IN ('failed','interrupted')", (now(), job_id)).rowcount
            if changed:
                db.conn.execute('UPDATE upload_files SET result=NULL,reason=NULL WHERE job_id=?', (job_id,))
            return bool(changed)

    def next_job(self):
        with RegistryDB(self.cfg.paths.registry_db) as db:
            row = db.conn.execute("SELECT job_id FROM upload_jobs WHERE status='queued' ORDER BY created_at,job_id LIMIT 1").fetchone()
            return row[0] if row else None

    def prune(self):
        with RegistryDB(self.cfg.paths.registry_db) as db, db.transaction():
            total = db.conn.execute('SELECT count(*) FROM upload_jobs').fetchone()[0]
            oldest = db.conn.execute("SELECT job_id FROM upload_jobs WHERE status IN ('done','failed','interrupted') "
                'ORDER BY created_at,job_id LIMIT ?', (max(0, total - self.cfg.api.upload.jobs_keep),)).fetchall()
            for row in oldest:
                db.conn.execute('DELETE FROM upload_jobs WHERE job_id=?', (row[0],))
        for row in oldest:
            shutil.rmtree(self.staging / row[0], ignore_errors=True)


class UploadManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.milvus = None
        self.store = UploadStore(cfg)
        self.submissions = asyncio.Lock()
        self.wake = asyncio.Event()
        self.stopping = False
        self.task = None

    async def start(self):
        await asyncio.to_thread(self.store.recover)
        self.task = asyncio.create_task(self.worker(), name='rag-upload-worker')

    async def stop(self):
        self.stopping = True
        self.wake.set()
        if self.task is not None:
            # 不取消 to_thread：先等入库线程释放 flock，再销毁 API 依赖。
            await self.task

    async def worker(self):
        while not self.stopping:
            self.wake.clear()
            job_id = await asyncio.to_thread(self.store.next_job)
            if job_id is None:
                await self.wake.wait()
                continue
            try:
                await asyncio.to_thread(self.execute, job_id)
            except IngestBusy:
                # 拿锁前不修改任务状态；CLI 占用时始终保持 queued。
                try:
                    await asyncio.wait_for(self.wake.wait(), self.cfg.api.upload.lock_retry_s)
                except TimeoutError:
                    pass
            except Exception as exc:
                log.warning('upload failed job_id=%s error_type=%s', job_id, type(exc).__name__)
                for f in await asyncio.to_thread(self.store.files, job_id):
                    if f['result'] is None:
                        await asyncio.to_thread(self.store.result, job_id, f['seq'], 'failed', '入库失败，请重试')
                await asyncio.to_thread(self.store.state, job_id, 'failed')
            await asyncio.to_thread(self.store.prune)

    def execute(self, job_id):
        from ragagent.ingest.pipeline import run_clean
        from ragagent.ingest.chunking.pipeline import run_chunk
        from ragagent.index.pipeline import run_index

        cfg, store = self.cfg, self.store
        with ingest_lock(cfg.paths.ingest_lock), RegistryDB(cfg.paths.registry_db) as db:
            files = store.files(job_id)
            root = Path(cfg.paths.uploads)
            root.mkdir(parents=True, exist_ok=True)
            selected = []
            for f in files:
                record = db.get(f['doc_id'])
                target = root / f['name']
                # 重试断点：已有当前版本成功入库的不再跑；parsing 保留批次继续。
                if (record and record['content_hash'] == f['content_hash'] and
                        record['status'] == 'indexed'):
                    store.result(job_id, f['seq'], 'unchanged')
                    continue
                staged = store.staging / job_id / str(f['seq'])
                if staged.is_file():
                    temp = root / ('.upload-' + uuid4().hex)
                    try:
                        shutil.copyfile(staged, temp)
                        os.replace(temp, target)
                    finally:
                        temp.unlink(missing_ok=True)
                elif not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != f['content_hash']:
                    store.result(job_id, f['seq'], 'failed', '上传文件缺失，请重新上传')
                    continue
                # 持锁时记录真正的旧版本，串行的两个覆盖任务不会把 created/updated 判错。
                f['previous_version'] = record['version'] if record else None
                selected.append(f)
            if selected:
                store.state(job_id, 'cleaning')
                run_clean(root, cfg=cfg, registry=db, source_prefix='@uploads',
                          source_names={f['name'] for f in selected})
                store.state(job_id, 'chunking')
                ready = set()
                for f in selected:
                    record = db.get(f['doc_id'])
                    if record and record['status'] in ('cleaned', 'indexed'):
                        try:
                            folder = Path(cfg.paths.clean) / f['doc_id'] / f"v{record['version']}"
                            doc = CleanDocument.model_validate_json((folder / 'clean.json').read_text(encoding='utf-8'))
                            chunks = run_chunk(doc, cfg=cfg.chunking)
                            output = Path(cfg.paths.chunks) / doc.doc_id / f'v{doc.version}.jsonl'
                            output.parent.mkdir(parents=True, exist_ok=True)
                            temp = output.with_suffix('.tmp')
                            try:
                                temp.write_text(''.join(c.model_dump_json() + '\n' for c in chunks), encoding='utf-8')
                                os.replace(temp, output)
                            finally:
                                temp.unlink(missing_ok=True)
                            ready.add(f['doc_id'])
                        except Exception as exc:
                            log.warning('upload chunk failed doc_id=%s error_type=%s', f['doc_id'], type(exc).__name__)
                            store.result(job_id, f['seq'], 'failed', '切分失败，请重试')
                    elif record and record['status'] == 'duplicate':
                        store.result(job_id, f['seq'], 'duplicate')
                        ready.add(f['doc_id'])  # 版本变成 duplicate 时清掉旧索引。
                    else:
                        store.result(job_id, f['seq'], 'failed', '清洗未完成，请重试')
                        if record and record['status'] in ('empty', 'unsupported'):
                            ready.add(f['doc_id'])
                if ready:
                    store.state(job_id, 'indexing')
                    run_index(cfg=cfg, registry=db, doc_ids=ready, client=self.milvus)
                results = {f['seq']: f for f in store.files(job_id)}
                for f in selected:
                    if results[f['seq']]['result'] is not None:
                        continue
                    record = db.get(f['doc_id'])
                    if record and record['status'] == 'indexed':
                        result = 'created' if f['previous_version'] is None else 'updated'
                        store.result(job_id, f['seq'], result)
                    else:
                        store.result(job_id, f['seq'], 'failed', '写索引失败，请重试')
            failed = any(f['result'] == 'failed' for f in store.files(job_id))
            store.state(job_id, 'failed' if failed else 'done')
