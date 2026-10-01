"""问答SSE、SQLite会话历史、登记表原文下载、L4反馈与指标挂载。

POST /api/chat接口定型依赖D-006；生产工厂：uvicorn ragagent.api.app:create_app --factory。
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from ragagent.api.sse import encode_event, graph_events
from ragagent.api.images import public_event, safe_filename
from ragagent.config import load_config
from ragagent.graph.build import build_graph
from ragagent.graph.deps import Deps
from ragagent.ingest.registry_db import RegistryDB
from ragagent.llm.client import close_clients
from ragagent.monitor.feedback import create_feedback_router
from ragagent.monitor.langfuse import build_client as build_trace_client, report_score, trace_config
from ragagent.monitor.metrics import MonitorMetrics
from ragagent.retrieval.filters import validate_filters

DOC_ID = re.compile(r'[0-9a-f]{16}')
SESSION_ID = re.compile(r'[A-Za-z0-9_-]{1,128}')
log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: StrictStr = Field(min_length=1)
    session_id: StrictStr | None = None
    filters: dict | None = None

    @field_validator('filters')
    @classmethod
    def safe_filters(cls, value):
        return validate_filters(value) or None


def session_id(request: Request, body_id=None) -> str:
    header = request.headers.get('X-Session-ID')
    if header is not None and body_id is not None and header != body_id:
        raise HTTPException(400, 'session_id 与 X-Session-ID 不一致')
    value = body_id if body_id is not None else header
    if not isinstance(value, str) or not SESSION_ID.fullmatch(value):
        raise HTTPException(400, '缺少或无效的 session_id')
    return value


class SessionLocks:
    def __init__(self):
        self.entries = {}

    @asynccontextmanager
    async def hold(self, sid):
        lock, users = self.entries.get(sid, (asyncio.Lock(), 0))
        self.entries[sid] = (lock, users + 1)
        try:
            async with lock:
                yield
        finally:
            _, users = self.entries[sid]
            if users == 1:
                del self.entries[sid]
            else:
                self.entries[sid] = (lock, users - 1)


def initial_state(question, filters) -> dict:
    # 每轮清空瞬时字段；messages/prev_intent保留在同session checkpoint。
    return dict(question=question, filters=filters, standalone_question='', queries=[], tried_queries=[],
                clarify_question=None, soft_filters=None, use_soft=False, scope_note=None,
                candidates=[], hits=[], expansions={}, retrieval={}, retry_count=0, context_blocks=[],
                attachments=[], answer='', gen_meta={})


def create_app(cfg=None, *, deps=None, graph=None, graph_factory=build_graph, metrics=None,
               trace_client=None, feedback_store=None) -> FastAPI:
    cfg = cfg or (deps.cfg if deps is not None else load_config())
    metrics = metrics or (deps.metrics if deps is not None else None) or MonitorMetrics(endpoints=cfg.llm.endpoints)
    sessions = SessionLocks()
    config_hash = hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()

    @asynccontextmanager
    async def lifespan(app):
        owned_milvus, owned_trace = None, None
        try:
            base = deps
            if base is None:
                from pymilvus import MilvusClient
                from ragagent.ingest.chunking.tokenizer import get_counter
                settings = cfg.services.milvus
                owned_milvus = MilvusClient(uri=settings.uri, user=os.getenv(settings.user_env, ''),
                                            password=os.getenv(settings.password_env, ''))
                base = Deps(cfg=cfg, count=get_counter(cfg.chunking.tokenizer), milvus=owned_milvus)
            base = replace(base, metrics=metrics)
            app.state.deps = base
            app.state.trace_client = trace_client if trace_client is not None else build_trace_client(cfg)
            owned_trace = app.state.trace_client if trace_client is None else None
            if graph is None:
                Path(cfg.paths.session_db).parent.mkdir(parents=True, exist_ok=True)
                async with AsyncSqliteSaver.from_conn_string(cfg.paths.session_db) as saver:
                    app.state.graph = graph_factory(saver)
                    yield
            else:
                app.state.graph = graph
                yield
        finally:
            await close_clients()
            if owned_milvus is not None:
                owned_milvus.close()
            if owned_trace is not None:
                try:
                    await asyncio.to_thread(owned_trace.flush)
                except Exception as exc:
                    log.warning('trace flush failed_type=%s', type(exc).__name__)

    app = FastAPI(lifespan=lifespan)
    app.state.metrics = metrics
    app.state.session_locks = sessions

    async def history_lookup(sid):
        snapshot = await app.state.graph.aget_state({'configurable': {'thread_id': sid}})
        return snapshot.values.get('messages', [])

    app.include_router(metrics.router(cfg.monitor.metrics.path))
    app.include_router(create_feedback_router(cfg, history_lookup, session_resolver=session_id,
        score_reporter=lambda sid, mid, rating: report_score(app.state.trace_client, message_id=mid, rating=rating),
        metrics=metrics, store=feedback_store))

    @app.post('/api/chat')
    async def chat(data: ChatRequest, request: Request):
        sid = session_id(request, data.session_id)
        if not data.question.strip():
            raise HTTPException(422, '问题不能为空')
        # 长度规范化交由L2 rules；API只避免无限body进入图。
        if len(data.question) > cfg.preprocess.max_question_chars:
            raise HTTPException(422, '问题超过长度上限')
        mid = uuid4().hex

        async def stream():
            route, status = 'rag', 'ok'
            def update(data):
                nonlocal route
                for values in data.values():
                    if isinstance(values, dict):
                        intent = values.get('intent')
                        scene = values.get('gen_meta', {}).get('scene')
                        if intent in ('rag', 'chat', 'clarify'):
                            route = intent
                        if scene == 'fallback':
                            route = 'fallback'
            async with sessions.hold(sid):
                registry = None
                try:
                    base = app.state.deps
                    if base.registry is None and Path(cfg.paths.registry_db).is_file():
                        registry = RegistryDB(cfg.paths.registry_db, readonly=True)
                    ctx = replace(base, registry=registry or base.registry, message_id=mid)
                    run_config = trace_config(cfg, session_id=sid, message_id=mid, config_hash=config_hash,
                                              intent='rag', client=app.state.trace_client)
                    run_config['configurable'] = {'thread_id': sid}
                    run_config['recursion_limit'] = 64
                    async for event in graph_events(app.state.graph, initial_state(data.question, data.filters),
                        run_config, ctx, request=request, on_update=update):
                        if event['type'] == 'error':
                            status = 'error'
                        yield encode_event(public_event(event))
                finally:
                    if registry is not None:
                        registry.close()
                    metrics.requests.labels(route=route, status=status).inc()
        return StreamingResponse(stream(), media_type='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no', 'X-Session-ID': sid})

    @app.get('/api/docs/{doc_id}/file')
    async def document_file(doc_id: str):
        if not DOC_ID.fullmatch(doc_id):
            raise HTTPException(400, '非法 doc_id')
        # 登记表只读/每请求连接，不让SQLite连接跨API worker线程。
        try:
            with RegistryDB(cfg.paths.registry_db, readonly=True) as db:
                row = db.get_document(doc_id)
        except (OSError, sqlite3.OperationalError):
            raise HTTPException(404, '文档不存在') from None
        if row is None or row['status'] in ('deleted', 'failed', 'unsupported'):
            raise HTTPException(404, '文档不存在')
        root = Path(cfg.paths.clean).resolve()
        target = (root / doc_id / f"v{row['version']}" / 'full.md').resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(404, '文档不存在')
        return FileResponse(target, media_type='text/markdown', filename=f'{doc_id}.md')

    @app.get('/api/images/{doc_id}/v{version}/{filename}')
    async def image_file(doc_id: str, version: int, filename: str):
        if not DOC_ID.fullmatch(doc_id) or version < 1 or not safe_filename(filename):
            raise HTTPException(400, '非法图片标识')
        try:
            with RegistryDB(cfg.paths.registry_db, readonly=True) as db:
                row = db.get_document(doc_id)
        except (OSError, sqlite3.OperationalError):
            raise HTTPException(404, '图片不存在') from None
        if row is None or row['status'] in ('deleted', 'failed', 'unsupported') or version > row['version']:
            raise HTTPException(404, '图片不存在')
        root = Path(cfg.paths.clean).resolve()
        target = (root / doc_id / f'v{version}' / 'images' / filename).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(404, '图片不存在')
        return FileResponse(target, headers={'X-Content-Type-Options': 'nosniff'})

    return app
