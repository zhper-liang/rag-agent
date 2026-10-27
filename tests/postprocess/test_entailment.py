import asyncio
from types import SimpleNamespace
import pytest
from ragagent.postprocess.entailment import ClaimSupport, CitationEntailment, verify_claims
from ragagent.postprocess.parse import Sentence


def test_numbers_require_support_from_the_specified_source(monkeypatch):
    async def call(*args):
        return CitationEntailment(claims=[
            ClaimSupport(index=0,supported=True,evidence_quote='k: int = 3'),
            ClaimSupport(index=1,supported=False,evidence_quote='')]),'main'
    monkeypatch.setattr('ragagent.postprocess.entailment.structured_call',call)
    sents=[Sentence(start=0,end=1,claim=f'默认生成{n}个候选动作',cites=[{'n':1,'state':'ok'}]) for n in [3,4]]
    blocks={1:SimpleNamespace(text='def expand_candidates(k: int = 3): pass')}
    cfg=SimpleNamespace(llm=SimpleNamespace(scenes=SimpleNamespace(eval=None),endpoints={}))
    assert asyncio.run(verify_claims(sents,blocks,'expand_candidates默认几个？',cfg))==2
    assert [s.cites[0]['state'] for s in sents]==['ok','unsupported']


def test_invalid_evidence_never_upgrades_citation(monkeypatch):
    async def call(*args):
        return CitationEntailment(claims=[ClaimSupport(index=0,supported=True,evidence_quote='编造证据')]),'main'
    monkeypatch.setattr('ragagent.postprocess.entailment.structured_call',call)
    s=Sentence(start=0,end=1,claim='默认生成3个',cites=[{'n':1,'state':'ok'}])
    cfg=SimpleNamespace(llm=SimpleNamespace(scenes=SimpleNamespace(eval=None),endpoints={}))
    asyncio.run(verify_claims([s],{1:SimpleNamespace(text='k=3')},'',cfg))
    assert s.cites[0]['state']=='unverified'
