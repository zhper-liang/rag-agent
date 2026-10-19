"""会话绑定的反馈路由与 SQLite 存储；reason 永不上传云端。"""
from __future__ import annotations

import asyncio
import inspect
import logging
import sqlite3
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StrictInt, field_validator

from ragagent.monitor.metrics import METRICS

log = logging.getLogger(__name__)


class FeedbackInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message_id: str
    rating: StrictInt
    reason: str | None = None

    @field_validator("rating")
    @classmethod
    def valid_rating(cls, value):
        if value not in {-1, 1}:
            raise ValueError("rating must be 1 or -1")
        return value

    @field_validator("message_id")
    @classmethod
    def nonempty(cls, value):
        if not value:
            raise ValueError("message_id is required")
        return value


class FeedbackStore:
    def __init__(self, path: str | Path, *, clock=time.time):
        self.path, self.clock = Path(path), clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY, message_id TEXT NOT NULL,
                session_id TEXT NOT NULL, rating INTEGER NOT NULL CHECK(rating IN (-1,1)),
                reason TEXT, created_at REAL NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS feedback_session_time ON feedback(session_id,created_at)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def save(self, session_id, data: FeedbackInput, limit: int):
        now = self.clock()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM feedback WHERE session_id=? AND created_at>?",
                               (session_id, now - 60)).fetchone()[0]
            if count >= limit:
                raise HTTPException(429, "该会话反馈过于频繁", headers={"Retry-After": "60"})
            db.execute("INSERT INTO feedback(message_id,session_id,rating,reason,created_at) VALUES(?,?,?,?,?)",
                       (data.message_id, session_id, data.rating, data.reason, now))

    def downvotes(self, since: float = 0) -> list[dict]:
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute("SELECT message_id,session_id,rating,reason,created_at "
                                               "FROM feedback WHERE rating=-1 AND created_at>=? ORDER BY created_at", (since,))]


async def _resolve(value):
    return await value if inspect.isawaitable(value) else value


def _message_id(message):
    if isinstance(message, dict):
        return message.get("message_id") or message.get("additional_kwargs", {}).get("message_id")
    return getattr(message, "additional_kwargs", {}).get("message_id")


def create_feedback_router(cfg, history_lookup, *, session_resolver=None, score_reporter=None,
                           metrics=METRICS, store=None) -> APIRouter:
    """history_lookup(session_id)->历史；session_resolver(Request)->当前会话 ID。

    两个回调可同步或异步。L3 应使用问答 API 相同的会话解析/checkpointer；
    默认仅接受 X-Session-ID，不从 feedback body 或 message_id 推导会话。
    score_reporter(session_id,message_id,rating) 仅接收分数，不接收 reason。
    """
    store = store if store is not None else FeedbackStore(cfg.paths.feedback_db)
    router = APIRouter()

    @router.post("/api/feedback")
    async def feedback(data: FeedbackInput, request: Request):
        session_id = await _resolve(session_resolver(request)) if session_resolver else request.headers.get("X-Session-ID")
        if not isinstance(session_id, str) or not session_id:
            raise HTTPException(400, "缺少当前会话")
        if data.reason is not None and len(data.reason) > cfg.monitor.feedback.reason_max_chars:
            raise HTTPException(422, "reason 超过长度上限")
        history = await _resolve(history_lookup(session_id))
        if not any(_message_id(m) == data.message_id for m in (history or [])):
            raise HTTPException(404, "message_id 不在该会话的历史里")
        # 反馈库是 rollback journal + BEGIN IMMEDIATE，写锁竞争会阻塞；放到线程里等。
        await asyncio.to_thread(store.save, session_id, data,
                                cfg.monitor.feedback.rate_limit_per_session_min)
        metrics.feedback.labels(rating=str(data.rating)).inc()
        if score_reporter:
            try:
                await _resolve(score_reporter(session_id, data.message_id, data.rating))
            except Exception as exc:
                log.warning("feedback score upload failed_type=%s", type(exc).__name__)
        return {"ok": True}

    return router


async def export_downvotes(store: FeedbackStore, history_lookup, *, since: float = 0) -> list[dict]:
    """每周可导出到本地：原问题、回答、引用状态和耗时。不上传 Langfuse。"""
    records = []
    for row in store.downvotes(since):
        history = await _resolve(history_lookup(row["session_id"])) or []
        for i, message in enumerate(history):
            if _message_id(message) != row["message_id"]:
                continue
            content = message.get("content", "") if isinstance(message, dict) else message.content
            meta = message.get("additional_kwargs", {}) if isinstance(message, dict) else message.additional_kwargs
            previous = history[i - 1] if i else {}
            question = previous.get("content", "") if isinstance(previous, dict) else previous.content
            records.append({**row, "question": question, "answer": content,
                            "citations": meta.get("post", {}), "timings": meta.get("timings", {})})
            break
    return records
