import asyncio
import json
from pathlib import Path

import pytest
import yaml

from eval.generate import GeneratedQuestion, bigram_jaccard, cosine, generate, sample_chunks, validate_question
from ragagent.config import Config

DOC = '0123456789abcdef'
QUOTE = '为保障安全且稳定运行，所有连接必须提前设置合理的等待时间。'


def config(**settings):
    cfg = Config.model_validate(yaml.safe_load(Path('config.yaml').read_text()))
    return cfg.model_copy(update={'eval': cfg.eval.model_copy(update={
        'generate': cfg.eval.generate.model_copy(update={'counts': {'single': 1}, **settings})})})


def chunk(cid='a', **updates):
    return dict(chunk_id=cid, doc_id=DOC, version=1, seq=0, tokens=100, text=QUOTE,
                searchable=True, type='text', **updates)


def question(kind='single', number=0, selected=None):
    selected = selected or [chunk()]
    answered = kind not in {'chat','clarify','unanswerable'}
    return {'tags': [], 'turns': ([{'question': '连接建议有哪些？', 'answer': '应提前配置等待时间。'}]
              if kind == 'multi_turn' else []) + [{'question': f'请解释这方面的运行建议第{number}项？'}],
            'expect': {'intent': kind if kind in {'chat','clarify'} else 'rag', 'answerable': answered,
                       'standalone': '连接建议的运行配置是什么？' if kind == 'multi_turn' else None,
                       'scope_doc_ids': [selected[0]['doc_id']] if kind == 'scoped' else []},
            'gold': [{'doc_id': c['doc_id'], 'version': c['version'], 'quote': c['text']} for c in
                     (selected if kind == 'multi_chunk' else selected[:1])] if answered else [],
            'reference': '需要事先设置连接等待时间。' if answered else ''}


async def unique_embed(texts):
    # 测试注入：高维确定性one-hot，不当成真实语义embedding。
    return [[1. if i == unique_embed.calls else 0. for i in range(40)] for _ in texts]


@pytest.fixture(autouse=True)
def reset_embed():
    unique_embed.calls = 0


async def embed(texts):
    result = await unique_embed(texts)
    unique_embed.calls += 1
    return result


async def retrieve(_):
    return {'retrieval': {'top_score': .1}}


def test_sample_stratified_latest_short_and_searchability():
    cs = [chunk('old'), {**chunk('new'), 'version': 2}, {**chunk('short'), 'tokens': 1},
          {**chunk('private'), 'doc_id': '1111111111111111', 'searchable': False},
          {**chunk('b'), 'doc_id': '2222222222222222'}, {**chunk('c'), 'doc_id': '3333333333333333'}]
    result = sample_chunks(cs, config().eval.generate, seed=42)
    assert {c['chunk_id'] for c in result} == {'new','b','c'}
    assert result == sample_chunks(cs, config().eval.generate, seed=42)


@pytest.mark.parametrize('change,reason', [
    ({'gold': [{'doc_id': DOC, 'version': 1, 'quote': '不存在的原文片段必须在二十个字以上才不会触发短句过滤。'}]}, 'quote_not_in_chunk'),
    ({'gold': [{'doc_id': DOC, 'version': 1, 'quote': '短句'}]}, 'short_quote'),
    ({'turns': [{'question': QUOTE}]}, 'copied_question'),
    ({'gold': []}, 'missing_gold'),
])
def test_filters(change, reason):
    q = GeneratedQuestion.model_validate({**question(), **change})
    assert validate_question(q, [chunk()], 'single', config().eval.generate) == reason


def test_type_rules_tags_scope_and_history():
    cfg = config()
    q = GeneratedQuestion.model_validate(question('multi_turn'))
    assert validate_question(q, [chunk()], 'multi_turn', cfg.eval.generate) is None
    q.turns[0].answer = None
    assert validate_question(q, [chunk()], 'multi_turn', cfg.eval.generate) == 'invalid_multi_turn'
    q = GeneratedQuestion.model_validate(question('scoped'))
    q.expect.scope_doc_ids = ['0000000000000000']
    assert validate_question(q, [chunk()], 'scoped', cfg.eval.generate) == 'invalid_scope'
    q = GeneratedQuestion.model_validate(question())
    assert validate_question(q, [{**chunk(), 'type':'image'}], 'single', cfg.eval.generate) == 'missing_image_tag'
    q.tags = ['image']
    assert validate_question(q, [{**chunk(), 'type':'image'}], 'single', cfg.eval.generate) is None


def test_regeneration_rejections_and_doc_limit():
    calls = []
    async def call(*args):
        calls.append(1)
        q = question(number=len(calls))
        if len(calls) == 1:
            return {'malformed': 1}, None
        if len(calls) == 2:
            q['turns'][0]['question'] = QUOTE
        return q, None
    cfg = config(counts={'single': 4}, max_per_doc=1)
    records, stats = asyncio.run(generate([chunk()], cfg, call=call, embed=embed, retrieve=retrieve))
    assert len(calls) == 3 and len(records) == 1
    assert stats['missing'] == {'single': 3}
    assert stats['rejections'] == {'invalid_schema':1, 'copied_question':1}
    assert 'chunk_id' not in records[0]['gold'][0]


def test_vectors_dedup_and_unanswerable_boundary():
    calls = []
    async def call(*_):
        calls.append(1)
        return question('unanswerable', len(calls)), None
    async def same(_): return [[1.,0.]]
    async def high(_): return {'retrieval': {'top_score': .3}}
    cfg = config(counts={'unanswerable':1})
    records, stats = asyncio.run(generate([chunk()], cfg, call=call, embed=same, retrieve=high))
    assert records == [] and len(calls) == 3 and stats['rejections']['has_answer'] == 3
    calls.clear()
    cfg = config(counts={'unanswerable':2})
    records, stats = asyncio.run(generate([chunk()], cfg, call=call, embed=same, retrieve=retrieve))
    assert len(records) == 1 and stats['rejections']['duplicate'] == 3
    with pytest.raises(ValueError): cosine([float('nan')],[1])
    with pytest.raises(ValueError): cosine([1],[1,2])
    assert bigram_jaccard(QUOTE,QUOTE) == 1


def test_all_seven_types_and_multi_chunk_gold():
    kinds = ['single','multi_chunk','unanswerable','multi_turn','chat','clarify','scoped']
    cs = [chunk(), {**chunk('b'), 'text': '另一节还要求按实际业务量配置资源，不要在高负载时临时扩大连接池。', 'seq':1}]
    async def call(scene, eps, schema, messages):
        prompt = json.loads(messages[-1].content)
        return question(prompt['type'], number=kinds.index(prompt['type']), selected=prompt['chunks']), None
    records, stats = asyncio.run(generate(cs, config(counts=dict.fromkeys(kinds,1), max_per_doc=10),
                                          call=call, embed=embed, retrieve=retrieve))
    assert {r['type'] for r in records} == set(kinds)
    assert not stats['missing']


def test_real_structured_call_with_local_http_stub(monkeypatch, llm_url, embed_url):
    from tests.stubs import llm_server
    from tests.llm.support import endpoints
    from ragagent.llm.client import close_clients
    from eval.embeddings import LocalEmbeddings
    monkeypatch.setenv('L3_STUB_KEY','stub')
    monkeypatch.setitem(llm_server.TOOL_ARGS, 'GeneratedQuestion', question)
    cfg = config()
    cfg = cfg.model_copy(update={'llm':cfg.llm.model_copy(update={'endpoints':endpoints(llm_url),
        'scenes':cfg.llm.scenes.model_copy(update={'eval':cfg.llm.scenes.eval.model_copy(update={'primary':'primary','fallback':'fallback'})})})})
    async def run():
        try:
            return await generate([chunk()], cfg, embed=LocalEmbeddings(embed_url, dimension=1024, timeout=2).aembed_documents,
                                  retrieve=retrieve)
        finally:
            await close_clients()
    records, stats = asyncio.run(run())
    assert len(records) == 1 and not stats['missing']
    import httpx
    calls = httpx.get(llm_url+'/_stub/calls').json()
    assert len(calls) == 1 and calls[0]['tool'] == 'GeneratedQuestion'


def test_cli_default_runtime_budget_includes_filter_and_saves_partial(monkeypatch, tmp_path):
    """保守2次/structured调用；出题+过滤合计，不凭预算耗尽伪造齐题。"""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from eval import generate as module
    from eval.embeddings import LocalEmbeddings
    from ragagent.graph.deps import Deps
    from ragagent.preprocess.schema import Preprocessed
    import eval.runtime
    import ragagent.llm.client
    import ragagent.retrieval.pipeline

    import ragagent.llm.observability as observability
    sentinel = object()
    monkeypatch.setattr(observability, 'METRICS', sentinel)
    cfg = config(counts={'single': 1, 'unanswerable': 2}, max_regen=0, max_per_doc=10)
    monkeypatch.setattr(module, 'load_config', lambda _: cfg)
    monkeypatch.setattr(LocalEmbeddings, 'from_config', lambda _: SimpleNamespace(aembed_documents=embed))
    events, calls = [], []

    @asynccontextmanager
    async def runtime(got_cfg, collection):
        assert got_cfg is cfg and collection == cfg.services.milvus.collection
        events.append('open')
        try:
            yield Deps(cfg=cfg, count=len)
        finally:
            events.append('runtime_closed')

    async def close():
        events.append('llm_closed')

    async def call(scene, eps, schema, messages):
        calls.append(schema.__name__)
        if schema is Preprocessed:
            return None, None  # 本地测试检索桩只关心调用是否被预算包装
        prompt = json.loads(messages[-1].content)
        return question(prompt['type'], number=len(calls)), None

    async def filtered(q, deps):
        assert deps.metrics is None and deps.write({'synthetic': True}) is None
        await deps.structured_call(cfg.llm.scenes.eval, cfg.llm.endpoints, Preprocessed, [])
        return {'retrieval': {'top_score': .1}}

    monkeypatch.setattr(eval.runtime, 'open_deps', runtime)
    monkeypatch.setattr(ragagent.llm.client, 'close_clients', close)
    monkeypatch.setattr(module, 'structured_call', call)
    monkeypatch.setattr(ragagent.retrieval.pipeline, 'run_pipeline', filtered)
    chunks, output = tmp_path/'chunks.jsonl', tmp_path/'questions.jsonl'
    chunks.write_text(json.dumps(chunk())+'\n')
    args = SimpleNamespace(config='config.yaml', chunks=str(chunks), output=str(output),
                           max_calls=6, deps_factory=None, seed=0)
    assert asyncio.run(module._main(args)) == 1
    assert observability.get_metrics() is sentinel
    assert calls == ['GeneratedQuestion', 'GeneratedQuestion', 'Preprocessed']
    assert len(output.read_text().splitlines()) == 2
    stats = json.loads(output.with_suffix('.generation.json').read_text())
    assert stats['reserved_physical_calls'] == 6
    assert stats['missing'] == {'unanswerable': 1}
    assert events == ['open', 'llm_closed', 'runtime_closed']


def test_cli_closes_llm_on_generation_error(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from eval import generate as module
    import ragagent.llm.client
    import ragagent.llm.observability as observability
    sentinel = object()
    monkeypatch.setattr(observability, 'METRICS', sentinel)
    events = []
    monkeypatch.setattr(module, 'load_config', lambda _: config())
    async def fail(*_, **__):
        raise RuntimeError('synthetic generation failure')
    async def close():
        events.append('closed')
    monkeypatch.setattr(module, 'generate', fail)
    monkeypatch.setattr(ragagent.llm.client, 'close_clients', close)
    path = tmp_path/'chunks.jsonl'
    path.write_text(json.dumps(chunk())+'\n')
    args = SimpleNamespace(config='config.yaml', chunks=str(path), output=str(tmp_path/'out.jsonl'),
                           max_calls=1, deps_factory=None, seed=0)
    with pytest.raises(RuntimeError, match='synthetic'):
        asyncio.run(module._main(args))
    assert events == ['closed']
    assert observability.get_metrics() is sentinel


def test_parallel_generation_reserves_doc_caps_and_resumes_without_extra_requests():
    """多个待决题争同一文档名额；恢复时保留已接受题而不会重新占用名额。"""
    active = peak = sequence = 0
    async def call(*args):
        nonlocal active, peak, sequence
        sequence += 1
        number = sequence
        active += 1; peak = max(peak, active)
        await asyncio.sleep(.01)
        active -= 1
        prompt = json.loads(args[-1][-1].content)
        return question(prompt['type'], number, prompt['chunks']), None
    cs = [{**chunk(str(i)), 'doc_id': f'{i:016x}'} for i in range(3)]
    cfg = config(counts={'single': 8}, max_per_doc=2, max_regen=0)
    records, stats = asyncio.run(generate(cs, cfg, call=call, embed=embed, retrieve=retrieve, concurrency=4))
    assert peak >= 2 and len(records) == 6 and sequence == 6
    assert stats['missing'] == {'single': 2}
    assert set(stats['per_doc'].values()) == {2}
    restored, second = asyncio.run(generate(cs, cfg, call=call, embed=embed, retrieve=retrieve,
                                           concurrency=4, resume_records=records))
    assert restored == records and sequence == 6
    assert second['resumed_records'] == 6 and second['missing'] == {'single': 2}
