"""D-003 A：StateGraph(context_schema=Deps)，节点 (state, runtime)。

依赖不进入 RAGState/checkpoint。Runtime.context 保存请求依赖；
Runtime.stream_writer 输出事件。测试可替换客户端、计数器及 write。
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from ragagent.config import Config
from ragagent.llm.structured import structured_call
from ragagent.llm.observability import get_metrics
from ragagent.prompt.env import build_env


@dataclass
class Deps:
    cfg: Config
    count: Callable[[str], int]
    clients: dict[tuple[str, str], Any] = field(default_factory=dict)
    sems: dict[str, asyncio.Semaphore] = field(default_factory=dict)
    write: Callable[[dict], None] | None = None
    milvus: Any | None = None
    embed_client: Any | None = None
    rerank_client: Any | None = None
    registry: Any | None = None
    structured_call: Callable = structured_call
    prompt_env: Any = None
    prompt_sampler: Any = None
    @property
    def metrics(self):
        """D-013 B：兼容节点读取；非dataclass字段，不接受构造/replace注入。"""
        return get_metrics()

    def __post_init__(self):
        if self.prompt_env is None:
            self.prompt_env = build_env(Path(self.cfg.prompt.dir) / self.cfg.prompt.version)


def emit(runtime, event: dict) -> None:
    """一个事件只写一次；API 的 custom stream 接收 Runtime.stream_writer。"""
    writer = runtime.context.write or runtime.stream_writer
    writer(event)
