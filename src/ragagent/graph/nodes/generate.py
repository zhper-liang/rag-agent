"""Prompt包与base64只在本节点驻留；失败不进入历史。"""
from __future__ import annotations

import logging
import time

from langgraph.runtime import Runtime

from ragagent.graph.deps import Deps, emit
from ragagent.graph.state import RAGState
from ragagent.llm.errors import GenerationFailed
from ragagent.llm.stream import stream_answer
from ragagent.prompt.assemble import assemble

log = logging.getLogger(__name__)


def build_gen_meta(pkg, result, info, count) -> dict:
    status = ('interrupted' if result.interrupted else
              'truncated' if result.finish_reason == 'length' else 'ok')
    return dict(status=status, scene=pkg.scene, prompt_version=pkg.version,
                est_input_tokens=pkg.est_input_tokens, prompt=pkg.meta,
                finish_reason=result.finish_reason,
                usage=result.usage or {'input_tokens': pkg.est_input_tokens, 'output_tokens': count(result.text)},
                usage_estimated=result.usage is None, **info)


async def generate_node(state: RAGState, runtime: Runtime[Deps]) -> dict:
    deps, started = runtime.context, time.perf_counter()
    pkg = assemble(state, deps.cfg, count=deps.count, env=deps.prompt_env)
    name = 'chat' if pkg.scene == 'chat' else 'generate'
    scene = getattr(deps.cfg.llm.scenes, name)
    emit(runtime, {'type': 'progress', 'stage': 'generating'})
    try:
        result, info = await stream_answer(pkg, scene, deps.cfg.llm.endpoints, deps.sems,
            lambda event: emit(runtime, event), clients=deps.clients, scene_name=name)
    except GenerationFailed as exc:
        log.warning('generation failed_type=%s', type(exc).__name__)
        emit(runtime, {'type': 'error', 'message': '回答服务暂时不可用，请稍后再试'})
        meta = dict(status='failed', scene=pkg.scene, prompt_version=pkg.version, attempts=exc.attempts)
    else:
        meta = build_gen_meta(pkg, result, info, deps.count)
    if deps.metrics is not None:
        metrics = deps.metrics
        metrics.stage_seconds.labels(stage='generate').observe(time.perf_counter() - started)
        attempts = meta.get('attempts', [])
        for i, attempt in enumerate(attempts):
            next_endpoint = attempts[i + 1]['endpoint'] if i + 1 < len(attempts) else meta.get('endpoint')
            outcome = 'retry' if next_endpoint == attempt['endpoint'] else 'failed'
            metrics.llm_calls.labels(scene=name, endpoint=attempt['endpoint'], outcome=outcome).inc()
        if meta['status'] != 'failed':
            outcome = 'fallback' if meta['fallback_used'] else 'ok'
            metrics.llm_calls.labels(scene=name, endpoint=meta['endpoint'], outcome=outcome).inc()
            for kind, key in [('input', 'input_tokens'), ('output', 'output_tokens')]:
                metrics.llm_tokens.labels(scene=name, endpoint=meta['endpoint'], kind=kind).inc(meta['usage'].get(key, 0))
            if meta.get('ttft_ms') is not None:
                metrics.stage_seconds.labels(stage='ttft').observe(meta['ttft_ms'] / 1000)
            if meta['images_dropped'] or pkg.meta.get('images_failed'):
                metrics.degraded.labels(kind='image').inc()
    return {'answer': result.text if meta['status'] != 'failed' else '', 'gen_meta': meta}
