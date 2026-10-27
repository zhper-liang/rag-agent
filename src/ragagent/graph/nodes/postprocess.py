from __future__ import annotations

from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from ragagent.graph.deps import Deps, emit
from ragagent.graph.state import RAGState
from ragagent.postprocess.final import build_final
from ragagent.postprocess.entailment import verify_claims
from ragagent.llm.observability import current_message_id
from ragagent.retrieval.http import post_json


async def postprocess_node(state: RAGState, runtime: Runtime[Deps]) -> dict:
    started = perf_counter()
    deps, cfg = runtime.context, runtime.context.cfg

    async def rerank_pairs(pairs):
        response = await post_json(deps.rerank_client, cfg, '/rerank/pairs',
            {'pairs': pairs, 'max_length': cfg.postprocess.rerank_max_length}, cfg.postprocess.verify_timeout_s)
        return response['scores']

    async def entailment(sents, blocks):
        return await verify_claims(sents, blocks, state['question'], cfg)

    event, plain, stats = await build_final(state, cfg, rerank_pairs, count=deps.count, message_id=current_message_id(), verify_claims=entailment)
    if event is None:
        return {}
    emit(runtime, event)
    if deps.metrics is not None:
        metrics = deps.metrics
        metrics.stage_seconds.labels(stage='postprocess').observe(perf_counter() - started)
        for citation in event['citations']:
            state_name = citation['state']
            metrics.citations.labels(state='invalid' if state_name in ('invalid_number', 'in_supplement') else 'valid').inc()
            if state_name in ('ok', 'unsupported'):
                metrics.citations.labels(state='supported' if state_name == 'ok' else 'unsupported').inc()
        for key in ('covered', 'uncovered'):
            metrics.citations.labels(state=key).inc(stats[key])
        if stats['verify_degraded']:
            metrics.degraded.labels(kind='verify').inc()
    meta = dict(state.get('gen_meta', {}), post=stats)
    return {'messages': [HumanMessage(content=state['question']), AIMessage(content=state['answer'],
        additional_kwargs=dict(plain=plain, message_id=event['message_id'], intent=state['intent'],
                               status=event['status'], cites=event['citations']))], 'gen_meta': meta}
