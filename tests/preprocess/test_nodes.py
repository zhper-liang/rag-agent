"""L2节点在真实Runtime和最小StateGraph下的契约与单轮评测入口。"""

import asyncio
import copy
import json

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from ragagent.graph.deps import Deps
from ragagent.graph.nodes.build_context import build_context
from ragagent.graph.nodes.clarify import clarify
from ragagent.graph.nodes.preprocess import preprocess
from ragagent.graph.nodes.rerank import rerank
from ragagent.graph.nodes.retrieve import retrieve
from ragagent.graph.nodes.rewrite import rewrite
from ragagent.graph.nodes.routes import route_after_preprocess, route_after_rerank, route_after_rewrite
from ragagent.graph.state import RAGState
from ragagent.ingest.registry_db import RegistryDB
from ragagent.retrieval.pipeline import run_pipeline
from tests.context.helpers import A, chunk, config, link
from tests.preprocess.test_preprocess import structured
from tests.retrieval.fakes import FakeMilvus
from tests.stubs.embed_server import create_app


def runtime(result=None, **kwargs):
    events = []
    deps = Deps(config(), len, structured_call=structured(result), write=events.append, **kwargs)
    return Runtime(context=deps), events


def test_preprocess_resets_all_per_turn_state_preserves_history_hard_and_prev_intent():
    async def run():
        rt, _ = runtime({'intent': 'clarify', 'standalone_question': '问题', 'clarify_question': '哪个？'})
        history = [HumanMessage('旧问'), AIMessage('旧答')]
        state = dict(question='本轮', messages=history, intent='clarify', prev_intent='chat',
                     retry_count=2, tried_queries=['old'], use_soft=False, soft_filters={'file_types': ['pdf']},
                     filters={'doc_ids': [A]}, scope_note='旧范围', clarify_question='旧反问',
                     candidates=[chunk()], hits=[chunk()], expansions={'old': []}, retrieval={'top_score': 1, 'dense_only': True},
                     context_blocks=[{'old': True}], attachments=[{'old': True}], answer='旧答', gen_meta={'old': True})
        before = copy.deepcopy(state)
        update = await preprocess(state, rt)
        assert state == before and 'messages' not in update
        assert update['prev_intent'] == 'clarify' and update['intent'] == 'rag'
        assert update['retry_count'] == 0 and update['tried_queries'] == [] and update['use_soft']
        assert update['filters'] == {'doc_ids': [A]} and update['soft_filters'] is None
        for key, empty in {'candidates': [], 'hits': [], 'expansions': {}, 'context_blocks': [],
                           'attachments': [], 'answer': '', 'gen_meta': {}}.items():
            assert update[key] == empty
        assert update['retrieval']['top_score'] == 0 and not update['retrieval']['dense_only']
        assert update['clarify_question'] is None and update['scope_note'] is None
        json.dumps(update)
    asyncio.run(run())


@pytest.mark.parametrize('question', ['', '字'*2001])
def test_rule_rejection_bypasses_llm_and_clarify_emits_one_token(question):
    async def run():
        rt, events = runtime()
        async def forbid(*args):
            raise AssertionError('不得调用LLM')
        rt.context.structured_call = forbid
        update = await preprocess({'question': question}, rt)
        assert update['intent'] == 'clarify' and update['queries'] == []
        answer = await clarify(update, rt)
        assert events[-1] == {'type': 'token', 'text': answer['answer']}
        assert sum(e['type'] == 'token' for e in events) == 1
    asyncio.run(run())


def test_invalid_hard_filter_rejected_before_llm():
    async def run():
        rt, _ = runtime()
        with pytest.raises(ValueError):
            await preprocess({'question': 'q', 'filters': {'doc_ids': ['bad" or true']}}, rt)
    asyncio.run(run())


def test_scope_conflict_drops_all_soft_and_retry_keeps_hard():
    async def run():
        rt, _ = runtime({'standalone_question': 'q', 'scope': {'file_types': ['PPT']}})
        update = await preprocess({'question': 'q', 'filters': {'file_types': ['pdf']}}, rt)
        assert update['soft_filters'] is None and update['scope_note'] is None
        rt.context.milvus = FakeMilvus([chunk(file_type='pdf')])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url='http://stub') as stub:
            rt.context.embed_client = stub
            state = {'queries': ['q'], 'filters': {'file_types': ['pdf']}, 'use_soft': False,
                     'soft_filters': {'file_types': ['ppt']}}
            await retrieve(state, rt)
            assert '"pdf"' in next(kwargs for kind, kwargs in rt.context.milvus.calls if kind == "hybrid")['reqs'][0].expr
            assert '"ppt"' not in next(kwargs for kind, kwargs in rt.context.milvus.calls if kind == "hybrid")['reqs'][0].expr
    asyncio.run(run())


def test_retrieve_rerank_context_preserve_summaries_budget_and_dict_contract():
    async def run():
        target, neighbor = link(chunk(0, text='部署服务需要配置端口'), chunk(1, searchable=False))
        fake = FakeMilvus([target, neighbor])
        rt, events = runtime({'standalone_question': '部署服务需要配置端口'}, milvus=fake)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url='http://stub') as stub:
            rt.context.embed_client = rt.context.rerank_client = stub
            state = {'question': '部署服务需要配置端口'}
            state.update(await preprocess(state, rt))
            state.update(await retrieve(state, rt))
            state.update(await rerank(state, rt))
            assert route_after_rerank(state, rt) == 'build_context'
            state.update(await build_context(state, rt))
            assert state['retrieval']['preprocess'] == {'degraded': False, 'intent': 'rag'}
            assert {'preprocess', 'embed', 'hybrid', 'fetch', 'rerank', 'expand', 'build_context'} <= state['retrieval']['timings_ms'].keys()
            assert state['retrieval']['context_usage']['total'] <= rt.context.cfg.context.budget_tokens
            assert neighbor['chunk_id'] in state['context_blocks'][0]['chunk_ids']
            assert not any(hasattr(item, 'model_dump') for item in state['hits'])
            json.dumps(state)
            assert events[-1] == {'type': 'meta', 'scope_note': None, 'attachments': []}
            assert [e['stage'] for e in events if e['type'] == 'progress'] == ['preprocessing', 'retrieving', 'reranking', 'building_context']
    asyncio.run(run())


@pytest.mark.parametrize('intent,route', [('rag', 'build_context'), ('chat', 'generate'), ('clarify', 'clarify')])
def test_quick_single_round_routes_no_generation_no_rewrite(intent, route):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url='http://stub') as stub:
            fake = FakeMilvus([chunk(text='完整问题')])
            calls = []
            async def call(scene, eps, schema, messages):
                calls.append(schema.__name__)
                return {'intent': intent, 'standalone_question': '完整问题', 'clarify_question': '请明确？'}, scene.primary
            deps = Deps(config(), len, milvus=fake, embed_client=stub, rerank_client=stub, structured_call=call)
            state = await run_pipeline('完整问题', deps)
            assert state['retrieval']['route'] == route and state['retrieval']['total_ms'] >= 0
            assert state['retry_count'] == 0 and state['answer'] == '' and calls == ['Preprocessed']
            assert isinstance(state['retrieval']['scope_doc_ids'], list)
            assert any(kind == "hybrid" for kind, _ in fake.calls) == (intent == 'rag')
            assert bool(state['context_blocks']) == (intent == 'rag')
    asyncio.run(run())


def test_quick_grade_failure_still_returns_one_round_context_and_prior_intent():
    async def run():
        fake = FakeMilvus([chunk(text='完全不同资料')])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url='http://stub') as stub:
            deps = Deps(config(), len, milvus=fake, embed_client=stub, rerank_client=stub,
                        structured_call=structured({'intent': 'clarify', 'standalone_question': '不存在查询'}))
            state = await run_pipeline('哪个', deps, messages=[HumanMessage('旧问'), AIMessage('哪个？', additional_kwargs={'intent': 'clarify'})])
            assert state['intent'] == 'rag' and state['prev_intent'] == 'clarify'
            assert state['retrieval']['grade'] == 'fail' and state['retrieval']['route'] == 'rewrite'
            assert state['retry_count'] == 0
    asyncio.run(run())


def test_real_langgraph_runtime_all_l2_nodes_and_conditional_routes():
    async def run():
        fake = FakeMilvus([chunk(text='部署手册正文')])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url='http://stub') as stub:
            deps = Deps(config(), len, milvus=fake, embed_client=stub, rerank_client=stub,
                        structured_call=structured({'standalone_question': '部署手册正文'}))
            g = StateGraph(RAGState, context_schema=Deps)
            for node in (preprocess, retrieve, rerank, build_context, clarify, rewrite):
                g.add_node(node.__name__, node)
            g.add_edge(START, 'preprocess')
            g.add_conditional_edges('preprocess', route_after_preprocess,
                                    {'retrieve': 'retrieve', 'clarify': 'clarify', 'generate': END})
            g.add_edge('retrieve', 'rerank')
            g.add_conditional_edges('rerank', route_after_rerank,
                                    {'build_context': 'build_context', 'rewrite': 'rewrite', 'fallback': END})
            g.add_conditional_edges('rewrite', route_after_rewrite, {'retrieve': 'retrieve', 'fallback': END})
            g.add_edge('build_context', END)
            g.add_edge('clarify', END)
            state = await g.compile().ainvoke({'question': '部署手册正文'}, context=deps)
            assert state['context_blocks'] and state['retrieval']['top_score'] == 1
            assert state['retry_count'] == 0
    asyncio.run(run())


def test_original_baseline_skips_llm_and_soft_scope_only_normalizes():
    from ragagent.retrieval.pipeline import run_original
    async def run():
        async def forbidden(*args):
            raise AssertionError('original不能调用LLM')
        fake = FakeMilvus([chunk(text='API 手册')])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url='http://stub') as stub:
            deps = Deps(config(), len, milvus=fake, embed_client=stub, rerank_client=stub, structured_call=forbidden)
            state = await run_original(' ＡＰＩ 手册 ', deps, filters={'doc_ids': [A]})
            assert state['queries'] == ['API 手册'] and state['soft_filters'] is None
            assert state['retrieval']['scope_doc_ids'] == [A]
            assert state['retrieval']['preprocess']['skipped'] and not state['retrieval']['preprocess']['degraded']
            assert state['retrieval']['mode'] == 'quick_single_round'
            assert state['context_blocks'] and state['answer'] == ''
    asyncio.run(run())


def test_injected_metrics_stages_hybrid_rerank_preprocess_image_and_retries(tmp_path):
    from ragagent.monitor.metrics import MonitorMetrics
    from ragagent.context.build import build_context_details
    async def run():
        metrics = MonitorMetrics()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url='http://stub') as stub:
            fake = FakeMilvus([chunk(text='q')], failures={'q': 'hybrid'})
            deps = Deps(config(), len, milvus=fake, embed_client=stub,
                        structured_call=structured(None), metrics=metrics)
            # 失败的rerank响应让融合结果直接进入context。
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)), base_url='http://bad') as bad:
                deps.rerank_client = bad
                state = await run_pipeline('q', deps)
                assert state['retrieval']['degraded'] and state['retrieval']['dense_only']
            for stage in ('preprocess', 'retrieve', 'rerank', 'build_context'):
                assert metrics.registry.get_sample_value('rag_stage_seconds_count', {'stage': stage}) == 1
            for kind in ('preprocess', 'hybrid', 'rerank'):
                assert metrics.registry.get_sample_value('rag_degraded_total', {'kind': kind}) == 1
            cfg = config(**{'paths.clean': str(tmp_path)})
            (tmp_path / 'broken.png').write_bytes(b'not an image')
            image = chunk(type='image', image_path='clean/broken.png')
            blocks, _, _ = build_context_details([image], {}, cfg, metrics=metrics)
            assert blocks[0].images[0].send_original is False
            assert metrics.registry.get_sample_value('rag_degraded_total', {'kind': 'image'}) == 1
            rt = Runtime(context=deps, stream_writer=lambda event: None)
            await rewrite({**state, 'soft_filters': {'file_types': ['pdf']}, 'use_soft': True}, rt)
            assert metrics.registry.get_sample_value('rag_retries_total') == 1
    asyncio.run(run())
