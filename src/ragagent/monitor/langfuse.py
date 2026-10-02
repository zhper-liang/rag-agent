"""Langfuse4.16 masked client/handler；无密钥、禁用、上传故障都不影响问答。"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, is_dataclass

from langfuse import Langfuse, propagate_attributes
from opentelemetry.trace import use_span
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
    """固定 SDK 4.16：根节点结束前纠正分类，异常只留下类别。

    根 observation 由该 handler 显式绑定，更新不依赖调用方当前 OTEL context。
    _runs / _otel_span 是固定 SDK 的适配边界，真实 exporter 测试覆盖。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._safe_root = None
        self._safe_metadata = {}
        self._safe_intent = None

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, metadata=None, **kwargs):
        if parent_run_id is None:
            self._safe_metadata = dict(metadata or {})
        result = super().on_chain_start(serialized, inputs, run_id=run_id,
                                        parent_run_id=parent_run_id, metadata=metadata, **kwargs)
        if parent_run_id is None:
            self._safe_root = self._runs.get(run_id)
            if self._safe_intent:
                self.update_intent(self._safe_intent)
        return result

    def update_intent(self, intent):
        try:
            if intent not in {"rag", "chat", "clarify"} or self._langfuse_client._mask is not mask:
                return False
            self._safe_intent = intent
            root = self._safe_root
            if root is None:
                return True  # on_update 可能先于 root callback；在 start/end 再应用
            if not root._otel_span.is_recording():
                return False
            metadata = {**self._safe_metadata, "intent": intent}
            root.update(metadata=metadata)
            # propagate_attributes 不经过 SDK mask；这里只传枚举白名单值。
            with use_span(root._otel_span, end_on_exit=False):
                with propagate_attributes(metadata={"intent": intent}):
                    pass
            return True
        except Exception as exc:
            log.warning("Langfuse intent update failed_type=%s", type(exc).__name__)
            return False

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        if parent_run_id is None:
            # on_update 可能等到 stream 队列消费才执行；final State 是最后保障。
            intent = outputs.get("intent") if isinstance(outputs, dict) else None
            if intent is not None:
                self.update_intent(intent)
        return super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    def on_chain_error(self, error, **kwargs):
        return super().on_chain_error(RuntimeError(type(error).__name__), **kwargs)

    def on_llm_error(self, error, **kwargs):
        return super().on_llm_error(RuntimeError(type(error).__name__), **kwargs)

    def on_tool_error(self, error, **kwargs):
        return super().on_tool_error(RuntimeError(type(error).__name__), **kwargs)


def update_trace_intent(run_config, *, intent):
    """L3 on_update 使用同一次 trace_config 的结果；禁用/无效/失败返回 False。

    不发额外 HTTP，不使用全局 handler，不接受任意 metadata 或原文。
    root 结束后的 hook 不可更新已结束 span；on_chain_end 已用 final intent 修正。
    """
    try:
        updated = False
        for handler in run_config.get("callbacks", []):
            if isinstance(handler, SafeCallbackHandler):
                updated = handler.update_intent(intent) or updated
        return updated
    except Exception as exc:
        log.warning("Langfuse intent hook failed_type=%s", type(exc).__name__)
        return False


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
