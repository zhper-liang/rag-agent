import asyncio

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.runtime import Runtime

from ragagent.api.app import initial_state
from ragagent.graph.build import build_graph
from ragagent.graph.deps import Deps, emit
from tests.integration.helpers import config


@pytest.mark.parametrize('case,expected', [
    ('rag', ['preprocess', 'retrieve', 'rerank', 'build_context', 'generate', 'postprocess']),
    ('chat', ['preprocess', 'generate', 'postprocess']),
    ('clarify', ['preprocess', 'clarify', 'postprocess']),
    ('degraded', ['preprocess', 'retrieve', 'rerank', 'build_context', 'generate', 'postprocess']),
    ('retry', ['preprocess', 'retrieve', 'rerank', 'rewrite', 'retrieve', 'rerank', 'build_context', 'generate', 'postprocess']),
    ('exhausted', ['preprocess', 'retrieve', 'rerank', 'rewrite', 'retrieve', 'rerank', 'rewrite', 'retrieve', 'rerank', 'fallback', 'postprocess']),
    ('empty_rewrite', ['preprocess', 'retrieve', 'rerank', 'rewrite', 'fallback', 'postprocess']),
])
def test_each_conditional_edge_with_runtime_and_sqlite(tmp_path, case, expected):
    async def run():
        cfg = config(tmp_path)
        seen, wire = [], []
        async def preprocess(state, runtime: Runtime[Deps]):
            seen.append('preprocess')
            return {'intent': case if case in ('chat', 'clarify') else 'rag', 'queries': ['q'], 'retry_count': 0}
        async def retrieve(state, runtime: Runtime[Deps]):
            seen.append('retrieve')
            return {}
        async def rerank(state, runtime: Runtime[Deps]):
            seen.append('rerank')
            passed = case == 'rag' or (case == 'retry' and state['retry_count'] == 1)
            return {'retrieval': {'top_score': 1 if passed else 0, 'degraded': case == 'degraded'}}
        async def rewrite(state, runtime: Runtime[Deps]):
            seen.append('rewrite')
            return {'retry_count': state['retry_count'] + 1, 'queries': [] if case == 'empty_rewrite' else ['q2']}
        async def build_context(state, runtime: Runtime[Deps]):
            seen.append('build_context')
            emit(runtime, {'type': 'meta', 'scope_note': None, 'attachments': []})
            return {}
        async def answer_node(state, runtime: Runtime[Deps]):
            node = 'clarify' if state['intent'] == 'clarify' else 'generate'
            seen.append(node)
            emit(runtime, {'type': 'token', 'text': '占位回答'})
            return {'answer': '占位回答', 'gen_meta': {'status': 'ok', 'scene': state['intent']}}
        async def fallback(state, runtime: Runtime[Deps]):
            seen.append('fallback')
            emit(runtime, {'type': 'token', 'text': 'fallback'})
            return {'answer': 'fallback', 'gen_meta': {'status': 'ok', 'scene': 'fallback'}}
        from ragagent.graph.nodes.postprocess import postprocess_node
        async def postprocess(state, runtime: Runtime[Deps]):
            seen.append('postprocess')
            return await postprocess_node(state, runtime)
        nodes = dict(preprocess=preprocess, retrieve=retrieve, rerank=rerank, rewrite=rewrite,
                     build_context=build_context, clarify=answer_node, generate=answer_node,
                     fallback=fallback, postprocess=postprocess)
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / 'graph.db')) as saver:
            graph = build_graph(saver, nodes=nodes)
            deps = Deps(cfg=cfg, count=len, write=wire.append)
            state = await graph.ainvoke(initial_state('q', None), context=deps,
                                       config={'configurable': {'thread_id': 's1'}})
            assert seen == expected
            assert wire[-1]['type'] == 'final' and len(state['messages']) == 2
            saved = await graph.aget_state({'configurable': {'thread_id': 's1'}})
            assert saved.values['messages'][-1].additional_kwargs['plain'] == state['answer']
            assert not (await graph.aget_state({'configurable': {'thread_id': 's2'}})).values
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / 'graph.db')) as saver:
            graph = build_graph(saver, nodes=nodes)
            assert len((await graph.aget_state({'configurable': {'thread_id': 's1'}})).values['messages']) == 2
    asyncio.run(run())
