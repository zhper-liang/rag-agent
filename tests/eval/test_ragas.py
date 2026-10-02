import asyncio
import json
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml
from fastapi import FastAPI, Request

from eval.ragas_eval import CallBudgetExceeded, RagasScorer, clean_answer
from ragagent.config import Config
from ragagent.llm.client import close_clients
from tests.llm.support import endpoints
from tests.stubs.serve import serve

QUESTION = '应该设置多长的连接等待时间？'
ANSWER = '连接等待设置为30秒。'


def cfg_for(base, embed_url):
    cfg = Config.model_validate(yaml.safe_load(Path('config.yaml').read_text()))
    u = urlparse(embed_url)
    return cfg.model_copy(update={
        'services':cfg.services.model_copy(update={'embed':cfg.services.embed.model_copy(update={'host':u.hostname,'port':u.port})}),
        'llm':cfg.llm.model_copy(update={'endpoints':endpoints(base),'scenes':cfg.llm.scenes.model_copy(update={
            'eval':cfg.llm.scenes.eval.model_copy(update={'primary':'primary','fallback':'fallback','timeout_s':5})})})})


def record():
    return {'case': {'id':'cn01','type':'single','turns':[{'question':QUESTION}],
                     'expect':{'answerable':True,'intent':'rag','standalone':None},
                     'gold':[{'doc_id':'0123456789abcdef','version':1,'quote':ANSWER}], 'reference':ANSWER},
            'state':{'answer':ANSWER+'[1]', 'context_blocks':[{'text':ANSWER}]}}


def scoring_stub():
    # 本线合成OpenAI协议桩；RAGAS自身的Prompt、解析、指标计算完全真实执行。
    app, calls = FastAPI(), []
    @app.post('/v1/chat/completions')
    async def completions(req: Request):
        body = await req.json()
        calls.append(body)
        tool = body.get('tools',[{}])[0].get('function',{}).get('name')
        if tool:
            payload = dict.fromkeys(('faithfulness','answer_relevancy','context_precision','context_recall','answer_correctness'),1.)
            if tool == 'NoAnswerDecision': payload = {'explicitly_no_answer': True}
            msg = {'role':'assistant','content':None,'tool_calls':[{'id':'t1','type':'function',
                   'function':{'name':tool,'arguments':json.dumps(payload)}}]}
        else:
            text = json.dumps(body['messages'],ensure_ascii=False)
            if 'NLIStatementOutput' in text:
                payload = {'statements':[{'statement':ANSWER,'reason':'直接记载','verdict':1}]}
            elif 'StatementGeneratorOutput' in text:
                payload = {'statements':[ANSWER]}
            elif 'ResponseRelevanceOutput' in text:
                payload = {'question':QUESTION,'noncommittal':0}
            elif 'ContextRecallClassifications' in text:
                payload = {'classifications':[{'statement':ANSWER,'reason':'直接记载','attributed':1}]}
            elif 'ClassificationWithReason' in text:
                payload = {'TP':[{'statement':ANSWER,'reason':'相同'}],'FP':[],'FN':[]}
            elif 'Verification' in text:
                payload = {'reason':'上下文相关','verdict':1}
            else:
                raise AssertionError('unrecognized RAGAS prompt schema')
            msg = {'role':'assistant','content':json.dumps(payload,ensure_ascii=False)}
        return {'id':'eval-stub','object':'chat.completion','created':0,'model':body['model'],
                'choices':[{'index':i,'message':msg,'finish_reason':'stop'} for i in range(body.get('n',1))],
                'usage':{'prompt_tokens':20,'completion_tokens':10,'total_tokens':30}}
    return app, calls


def test_strip_parsed_supplement_and_only_noncode_citations():
    heading = '补充说明'
    answer = '事实[1]和[20]。\n`数组[1]`\n```python\na[2]\n```\n## 补充说明\n不应送给RAGAS[3]'
    cleaned = clean_answer(answer, heading)
    assert '事实和。' in cleaned and '`数组[1]`' in cleaned and 'a[2]' in cleaned
    assert '不应送给' not in cleaned and '补充说明' not in cleaned


def test_real_ragas_all_five_metrics_local_http(monkeypatch, embed_url):
    monkeypatch.setenv('L3_STUB_KEY','stub')
    app,calls = scoring_stub()
    with serve(app) as base:
        async def run():
            scorer = RagasScorer(cfg_for(base,embed_url),max_calls=30)
            try:
                r=record()
                scores = await scorer(r['case'],r['state'])
                assert set(scores) == {'faithfulness','answer_relevancy','context_precision','context_recall','answer_correctness'}
                assert all(v == pytest.approx(1.) for v in scores.values())
                assert 7 <= scorer.budget.used <= 15
                return scores
            finally:
                await close_clients()
        asyncio.run(run())
    assert calls and all(c['model']=='primary' and not c.get('stream') for c in calls)
    assert all(c.get('temperature')==0 for c in calls)


def test_budget_exhaustion_and_noanswer_skip(monkeypatch, embed_url):
    monkeypatch.setenv('L3_STUB_KEY','stub')
    app,calls = scoring_stub()
    with serve(app) as base:
        async def run():
            try:
                scorer=RagasScorer(cfg_for(base,embed_url),max_calls=1,metric_names=('faithfulness',))
                r=record()
                with pytest.raises(CallBudgetExceeded): await scorer(r['case'],r['state'])
                assert scorer.budget.used == 1
                r['case']['expect']['answerable']=False
                assert await scorer(r['case'],r['state']) == {}
            finally:
                await close_clients()
        asyncio.run(run())
    assert len(calls)==1


def test_chinese_comparison_and_explicit_noanswer_judge(monkeypatch, embed_url):
    monkeypatch.setenv('L3_STUB_KEY','stub')
    app,calls=scoring_stub()
    with serve(app) as base:
        async def run():
            try:
                scorer=RagasScorer(cfg_for(base,embed_url),max_calls=50)
                with pytest.raises(ValueError,match='30'): await scorer.validate_chinese([record()])
                assert not calls
                english=record();english['case']['turns'][-1]['question']='What is the answer?'
                english['case']['expect'].pop('standalone',None)
                with pytest.raises(ValueError,match='Chinese user questions'):
                    await scorer.validate_chinese([english],count=1)
                duplicate=record()
                with pytest.raises(ValueError,match='distinct question ids'):
                    await scorer.validate_chinese([record(),duplicate],count=2)
                assert not calls
                other=record();other['case']['id']='cn02'
                report=await scorer.validate_chinese([record(),other],count=2)
                assert report['samples']==2 and not report['needs_prompt_adaptation']
                assert await scorer.judge_no_answer(record()['case'],{'answer':'资料没有答案'}) is True
            finally:
                await close_clients()
        asyncio.run(run())


def test_multisampling_uses_three_independent_n1_requests(monkeypatch, embed_url):
    from fastapi.responses import JSONResponse
    from langchain_core.prompt_values import StringPromptValue
    from eval.calls import counted_calls, current_summary
    monkeypatch.setenv('L3_STUB_KEY','stub')
    app = FastAPI()
    requests = []
    @app.post('/v1/chat/completions')
    async def completion(request: Request):
        body = await request.json()
        requests.append(body)
        if body.get('n',1) != 1:
            return JSONResponse({'error':{'message':'n is not supported'}},status_code=400)
        return {'id':'s','object':'chat.completion','created':0,'model':'primary',
                'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':f'sample-{len(requests)}'}}]}
    with serve(app) as base:
        async def run():
            scorer = RagasScorer(cfg_for(base,embed_url), max_calls=3)
            try:
                with counted_calls(3):
                    result = await scorer.llm.agenerate_text(StringPromptValue(text='public sample'),n=3)
                    assert current_summary()['actual_calls'] == 3
                assert [g.text for g in result.generations[0]] == ['sample-1','sample-2','sample-3']
                assert scorer.budget.used == 3
            finally:
                await close_clients()
        asyncio.run(run())
    assert len(requests) == 3 and all(r.get('n') == 1 for r in requests)
