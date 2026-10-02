"""结构化单次调用（大模型生成 §6.2；D-005 A）。"""
from __future__ import annotations

import asyncio
import logging

from ragagent.llm.client import build_client, endpoint_semaphore
from ragagent.llm.observability import get_metrics, structured_scene

log = logging.getLogger(__name__)


async def structured_call(scene, eps, schema, messages):
    """主备各尝试一次；失败/超时/None/校验失败均切换，最终返回 (None, None)。

取消直接向上传播。日志只含接入点名称和异常类型，绝不记录消息正文。
"""
    metrics, scene_name = get_metrics(), structured_scene(schema)
    for name in (scene.primary, scene.fallback):
        ep = eps[name]
        try:
            async def invoke():
                async with endpoint_semaphore(ep):
                    llm = build_client(ep, scene).with_structured_output(
                        schema, method=ep.structured_output
                    )
                    result = await llm.ainvoke(messages)
                    if result is None:
                        raise ValueError("structured output is None")
                    if hasattr(schema, "model_validate"):
                        result = schema.model_validate(result)
                    return result

            result = await asyncio.wait_for(invoke(), timeout=scene.timeout_s)
            if metrics is not None:
                metrics.llm_calls.labels(scene=scene_name, endpoint=name,
                                         outcome="ok" if name == scene.primary else "fallback").inc()
            return result, name
        except Exception as exc:
            if metrics is not None:
                metrics.llm_calls.labels(scene=scene_name, endpoint=name, outcome="failed").inc()
            log.warning("structured_call endpoint=%s failed_type=%s", name, type(exc).__name__)
    return None, None
