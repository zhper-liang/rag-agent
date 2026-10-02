"""固定RAGAS 0.4.3接口；本地embedding、受限评测LLM、中文直接评分对照。"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
from pathlib import Path

# RAGAS产品遥测也关闭；评测内容不得通过Langfuse上传。
os.environ['RAGAS_DO_NOT_TRACK'] = 'true'

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from ragas import EvaluationDataset, SingleTurnSample, aevaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (AnswerCorrectness, Faithfulness, LLMContextPrecisionWithReference,
                           LLMContextRecall, ResponseRelevancy)
from ragas.run_config import RunConfig

from eval.calls import counted_calls, current_summary
from eval.embeddings import LocalEmbeddings
from ragagent.config import load_config
from ragagent.llm.client import build_client, close_clients, endpoint_semaphore
from ragagent.llm.structured import structured_call

NAMES = ('faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness')


class CallBudgetExceeded(RuntimeError):
    pass


class CallBudget:
    def __init__(self, limit):
        if limit < 1:
            raise ValueError('LLM call budget must be positive')
        self.limit, self.used = limit, 0

    def reserve(self, calls=1):
        if self.used + calls > self.limit:
            raise CallBudgetExceeded('eval LLM call budget exhausted')
        self.used += calls


class EvalLLM(LangchainLLMWrapper):
    """继续使用L3客户端/接入点并发限制；禁SDK与RAGAS的隐式重试。"""
    def __init__(self, cfg, budget, run_config):
        self.cfg, self.budget = cfg, budget
        super().__init__(build_client(cfg.llm.endpoints[cfg.llm.scenes.eval.primary], cfg.llm.scenes.eval),
                         run_config=run_config)

    async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        if not isinstance(n, int) or n < 1:
            raise ValueError('generation count must be positive')
        # OpenAI-compatible 接入点可能仅支持 n=1；RAGAS 的 n 个样本必须独立真实采样。
        # 不复制一个回答冒充多样本；每次采样的主备调用均计入物理及预约预算。
        scene = self.cfg.llm.scenes.eval
        results = []
        for _ in range(n):
            for index, name in enumerate((scene.primary, scene.fallback)):
                self.budget.reserve()
                ep = self.cfg.llm.endpoints[name]
                client = build_client(ep, scene)
                try:
                    async with endpoint_semaphore(ep):
                        result = await client.agenerate_prompt([prompt], n=1, temperature=scene.temperature,
                                                               stop=stop, callbacks=[])
                    results.append(result)
                    break
                except Exception:
                    if index:
                        raise
        return results[0].model_copy(update={
            'generations': [[generation for result in results for generation in result.generations[0]]],
            'llm_output': None})


class DirectScores(BaseModel):
    faithfulness: float = Field(ge=0, le=1)
    answer_relevancy: float = Field(ge=0, le=1)
    context_precision: float = Field(ge=0, le=1)
    context_recall: float = Field(ge=0, le=1)
    answer_correctness: float = Field(ge=0, le=1)


class NoAnswerDecision(BaseModel):
    explicitly_no_answer: bool


def clean_answer(answer, heading):
    """复用L3等长代码mask和supplement解析；代码内[数字]保留。"""
    from ragagent.postprocess.parse import CITE, find_supplement, mask_code
    masked = mask_code(answer)
    supplement = find_supplement(masked, heading)
    end = len(answer) if supplement is None else supplement
    text = answer[:end]
    for match in reversed(list(CITE.finditer(masked[:end]))):
        text = text[:match.start()] + text[match.end():]
    return text.strip()


def sample(case, state, cfg):
    return SingleTurnSample(user_input=case['expect'].get('standalone') or case['turns'][-1]['question'],
                            response=clean_answer(state.get('answer', ''), cfg.prompt.supplement_heading),
                            retrieved_contexts=[b['text'] for b in state.get('context_blocks', [])],
                            reference=case['reference'])


class RagasScorer:
    def __init__(self, cfg, *, max_calls, metric_names=NAMES, prompt_dir=None, language='chinese'):
        self.cfg, self.budget = cfg, CallBudget(max_calls)
        self.run_config = RunConfig(timeout=cfg.llm.scenes.eval.timeout_s, max_retries=1, max_workers=1)
        self.llm = EvalLLM(cfg, self.budget, self.run_config)
        self.embeddings = LangchainEmbeddingsWrapper(LocalEmbeddings.from_config(cfg), run_config=self.run_config)
        self.metric_names, self.prompt_dir, self.language = metric_names, prompt_dir, language

    def metrics(self):
        result = []
        for name, ctor in zip(NAMES, (Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference,
                                     LLMContextRecall, AnswerCorrectness)):
            if name in self.metric_names:
                metric = ctor()
                if self.prompt_dir:
                    metric.set_prompts(**metric.load_prompts(self.prompt_dir, language=self.language))
                result.append(metric)
        if not result:
            raise ValueError('no RAGAS metrics selected')
        return result

    async def __call__(self, case, state):
        if not case['expect']['answerable'] or not case.get('gold'):
            return {}  # 无答案题不跑RAGAS
        row = sample(case, state, self.cfg)
        if not row.response or not row.retrieved_contexts:
            raise ValueError('answerable RAGAS sample requires an actual answer and contexts')
        result = await aevaluate(EvaluationDataset(samples=[row]), metrics=self.metrics(), llm=self.llm,
                                # timeout_s 属于每次接入点请求；单个metric可串行消费多份context
                                # 和主备。保留单请求60s，在metric层按实际context数量给总时限。
                                embeddings=self.embeddings, run_config=RunConfig(
                                    timeout=self.cfg.llm.scenes.eval.timeout_s * max(6, 2*len(row.retrieved_contexts)),
                                    max_retries=1, max_workers=1), callbacks=[],
                                raise_exceptions=True, show_progress=False)
        scores = {}
        for key, value in result.scores[0].items():
            normalized = {'answer_relevance':'answer_relevancy', 'llm_context_precision_with_reference':'context_precision'}.get(key,key)
            if normalized not in self.metric_names:
                continue
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f'RAGAS metric {key} is missing/nonfinite')
            scores[normalized] = value
        if set(scores) != set(self.metric_names):
            raise ValueError(f'missing RAGAS metrics: {set(self.metric_names)-set(scores)}')
        return scores

    async def _structured(self, schema, messages):
        self.budget.reserve(2)  # 主备最多两个物理请求；保守预留，不能超限
        parsed, _ = await structured_call(self.cfg.llm.scenes.eval, self.cfg.llm.endpoints, schema, messages)
        if parsed is None:
            raise RuntimeError(f'eval judge {schema.__name__} failed on both endpoints')
        return parsed

    async def judge_no_answer(self, case, state):
        decision = await self._structured(NoAnswerDecision, [
            SystemMessage('只判断回答是否明确说明资料没有答案或无法从资料确认。猜测、编造答案不算。'),
            HumanMessage(json.dumps({'question': case['turns'][-1]['question'], 'answer': state.get('answer','')}, ensure_ascii=False))])
        return decision.explicitly_no_answer

    async def direct_scores(self, case, state):
        row = sample(case, state, self.cfg)
        scores = await self._structured(DirectScores, [
            SystemMessage('按0到1评分：faithfulness=回答断言中由上下文支撑的比例；answer_relevancy=对问题的相关程度；'
                          'context_precision=上下文的有效检索精确率；context_recall=参考答案内容被上下文覆盖的比例；'
                          'answer_correctness=回答相对参考答案的事实及语义正确性。只评资料内正文。'),
            HumanMessage(row.model_dump_json())])
        return scores.model_dump()

    async def validate_chinese(self, samples, *, count=None, max_difference=.15, concurrency=1, on_result=None):
        count = self.cfg.eval.ragas_check_samples if count is None else count
        rows = [r for r in samples if r['case']['expect']['answerable'] and r['case'].get('gold')]
        if len({r['case']['id'] for r in rows}) != len(rows):
            raise ValueError('Chinese validation needs distinct question ids')
        if any(not re.search(r'[\u3400-\u9fff]', sample(r['case'], r['state'], self.cfg).user_input or '') for r in rows):
            raise ValueError('Chinese validation requires Chinese user questions')
        if len(rows) < count:
            raise ValueError(f'Chinese validation needs {count} answerable samples, only {len(rows)} provided')
        if concurrency < 1:
            raise ValueError('Chinese validation concurrency must be positive')
        sem = asyncio.Semaphore(concurrency)
        completed = 0
        async def one(r):
            nonlocal completed
            async with sem:
                try:
                    measured = await self(r['case'], r['state'])
                    direct = await self.direct_scores(r['case'], r['state'])
                    result = {'id': r['case']['id'], 'status': 'ok', 'ragas': measured, 'direct': direct}
                except Exception as exc:
                    result = {'id': r['case']['id'], 'status': 'failed', 'error_type': type(exc).__name__}
                completed += 1
                if on_result is not None:
                    on_result(result)
                print(f"chinese={completed}/{count} id={r['case']['id']} status={result['status']}", flush=True)
                return result
        results = await asyncio.gather(*(one(r) for r in rows[:count]))
        valid = [r for r in results if r['status'] == 'ok']
        mae = {k: sum(abs(r['ragas'][k]-r['direct'][k]) for r in valid)/len(valid) if valid else None for k in self.metric_names}
        return {'samples':count, 'mean_absolute_difference':mae, 'max_difference':max_difference,
                'needs_prompt_adaptation':any(v is None or v > max_difference for v in mae.values()),
                'successful_samples':len(valid), 'missing':count-len(valid),
                'reserved_calls':self.budget.used, 'cases':results}


async def _main(args):
    with counted_calls(args.max_calls, ledger=Path(args.output).with_suffix('.calls.json')):
        return await _main_open(args)


async def _main_open(args):
    cfg = load_config(args.config)
    samples = [json.loads(s) for s in Path(args.input).read_text().splitlines() if s.strip()]
    scorer = RagasScorer(cfg, max_calls=args.max_calls, prompt_dir=args.prompt_dir)
    progress = []
    def persist(result):
        progress.append(result)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({'status':'running', 'requested':cfg.eval.ragas_check_samples,
            'completed':len(progress), 'cases':progress, **current_summary()}, ensure_ascii=False, indent=2)+'\n')
    try:
        if args.check_chinese:
            report = await scorer.validate_chinese(samples, max_difference=args.max_difference,
                concurrency=args.concurrency, on_result=persist)
        else:
            report = {'cases':[{'id':r['case']['id'], 'scores':await scorer(r['case'],r['state'])} for r in samples],
                      'reserved_calls':scorer.budget.used}
        report.update(current_summary())
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        print(f'samples={len(samples)} reserved_calls={scorer.budget.used} output={args.output}')
        return int(report.get('needs_prompt_adaptation', False) or report.get('missing', 0) > 0)
    finally:
        await close_clients()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--input', required=True, help='本地JSONL记录，每行为 {case:评测题,state:实际RAGState}')
    p.add_argument('--output', required=True)
    p.add_argument('--max-calls', type=int, required=True)
    p.add_argument('--check-chinese', action='store_true')
    p.add_argument('--max-difference', type=float, default=.15, help='直接评分对照MAE的人工审查阈值')
    p.add_argument('--prompt-dir', help='经检查的RAGAS中文版prompt文件目录')
    return asyncio.run(_main(p.parse_args(argv)))


if __name__ == '__main__':
    raise SystemExit(main())
