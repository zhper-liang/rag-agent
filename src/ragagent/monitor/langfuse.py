"""Langfuse4.16 masked client/handler；无密钥、禁用、上传故障都不影响问答。"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, is_dataclass

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

log = logging.getLogger(__name__)
KEEP = {"chunk_id", "doc_id", "version", "seq", "type", "intent", "status", "endpoint",
        "model", "prompt_version", "finish_reason", "cite", "state", "stage", "scene"}


def _mask(x, key=None):
    if hasattr(x, "model_dump"):
        x = x.model_dump(mode="json")
    elif is_dataclass(x) and not isinstance(x, type):
        x = asdict(x)
    if isinstance(x, dict):
        return {k: _mask(v, k) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_mask(v, key) for v in x]
    if isinstance(x, str):
        return x if key in KEEP else f"[masked:{len(x)}]"
    if x is None or isinstance(x, (bool, int, float)):
        return x
    # 非 JSON 对象不得透传 repr（可能含 prompt）；bytes 同样遮掉。
    return f"[masked:{len(x) if isinstance(x, bytes) else len(str(x))}]"


def mask(*, data, **_):
    return _mask(data)


class SafeCallbackHandler(CallbackHandler):
    """SDK status_message 可能直接使用 str(error)，送入前只留下异常类别。"""
    def on_chain_error(self, error, **kwargs):
        return super().on_chain_error(RuntimeError(type(error).__name__), **kwargs)

    def on_llm_error(self, error, **kwargs):
        return super().on_llm_error(RuntimeError(type(error).__name__), **kwargs)

    def on_tool_error(self, error, **kwargs):
        return super().on_tool_error(RuntimeError(type(error).__name__), **kwargs)


def build_client(cfg, *, env=None, **sdk_options):
    """仅启用 masked 云端：协议禁止本线选择 unmasked 客户端。

    SDK 按 public_key 复用资源，若同公钥先被其他代码无脱敏初始化，拒绝使用。
    raw prompt 抽样需要另行合规裁决；此接口绝不因 sample_rate 而绕过 mask。
    """
    settings = cfg.monitor.langfuse
    if not settings.enabled:
        return None
    if not settings.mask:
        log.warning("Langfuse disabled: unmasked cloud export is not permitted")
        return None
    env = os.environ if env is None else env
    public, secret = env.get(settings.public_key_env), env.get(settings.secret_key_env)
    if not public or not secret:
        log.warning("Langfuse disabled: missing credentials")
        return None
    # 不允许 sdk_options 覆盖白名单回调或密钥。
    if {"mask", "public_key", "secret_key", "host", "base_url"} & sdk_options.keys():
        raise ValueError("cannot override masked client configuration")
    try:
        client = Langfuse(public_key=public, secret_key=secret, base_url=settings.host,
                          mask=mask, **sdk_options)
        if client._mask is not mask:
            log.warning("Langfuse disabled: existing SDK client has an unsafe mask")
            return None
        return client
    except Exception as exc:
        log.warning("Langfuse initialization failed_type=%s", type(exc).__name__)
        return None


def get_handler(cfg, *, message_id=None, env=None, client=None):
    client = client if client is not None else build_client(cfg, env=env)
    if client is None or client._mask is not mask:
        return None
    try:
        public_key = (os.environ if env is None else env).get(cfg.monitor.langfuse.public_key_env)
        if not public_key:
            return None
        trace_context = {"trace_id": Langfuse.create_trace_id(seed=message_id)} if message_id else None
        handler = SafeCallbackHandler(public_key=public_key, trace_context=trace_context)
        if handler._langfuse_client._mask is not mask:
            return None
        return handler
    except Exception as exc:
        log.warning("Langfuse handler initialization failed_type=%s", type(exc).__name__)
        return None


def trace_config(cfg, *, session_id, message_id, config_hash, intent, client=None, env=None) -> dict:
    """提供给 graph.ainvoke 的 RunnableConfig；无 handler 时空 callbacks。

    session_id 用 SDK 保留的关联字段；资料只通过 mask 处理的 observations 上报。
    """
    handler = get_handler(cfg, message_id=message_id, client=client, env=env)
    return {"callbacks": [handler] if handler else [], "metadata": {
        "langfuse_session_id": session_id,
        "message_id": message_id, "config_hash": config_hash,
        "prompt_version": cfg.prompt.version, "intent": intent}}


def report_score(client, *, message_id, rating):
    """挂到 message_id 对应 deterministic trace，只传分数，不接受用户 reason。"""
    if client is None:
        return
    try:
        client.create_score(name="feedback", value=rating, data_type="NUMERIC",
                            trace_id=Langfuse.create_trace_id(seed=message_id))
    except Exception as exc:
        log.warning("Langfuse feedback failed_type=%s", type(exc).__name__)
