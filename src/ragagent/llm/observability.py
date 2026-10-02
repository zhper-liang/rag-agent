"""D-013 B：进程全局指标入口，LLM/Deps 不增加监控注入参数。

每个服务进程只运行一个活动 API 应用。测试通过 monkeypatch 替换 METRICS；
离线评测可用 metrics_scope(None) 暂停指标，退出（包括异常）恢复原对象。
message_id 与结构化场景取 RunnableConfig 元数据，不进入 Deps/checkpoint。
"""
from __future__ import annotations

from contextlib import contextmanager

from langgraph.config import get_config

METRICS = None


def get_metrics():
    return METRICS


@contextmanager
def metrics_scope(metrics):
    """仅用于进程生命周期/离线批次，不是并发请求级的隔离上下文。"""
    global METRICS
    previous, METRICS = METRICS, metrics
    try:
        yield
    finally:
        METRICS = previous


def runnable_metadata() -> dict:
    try:
        return get_config().get('metadata', {})
    except RuntimeError:  # 直接调用节点/结构化函数的本地测试没有Runnable上下文。
        return {}


def structured_scene(schema) -> str:
    node = runnable_metadata().get('langgraph_node')
    if node in ('preprocess', 'rewrite'):
        return node
    return {'Preprocessed': 'preprocess', 'Rewrite': 'rewrite'}.get(
        getattr(schema, '__name__', ''), 'eval')


def current_message_id() -> str | None:
    value = runnable_metadata().get('message_id')
    return value if isinstance(value, str) and value else None
