"""采集真实校准观测；生产索引只读，仅在 l4_ 临时集合实验切分。"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import httpx
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage

from eval.align import align_gold
from eval.calls import counted_calls, current_summary
from eval.calibrate import WEIGHTS, CHUNK_TARGETS
from eval.embeddings import LocalEmbeddings
from eval.runtime import open_deps
from ragagent.config import load_config
from ragagent.llm.observability import metrics_scope


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


async def token_samples(cfg, chunks, data, persist, *, size, concurrency):
    """相同 messages 模板的空正文请求测固定开销，逐chunk读取真实usage。"""
    ep = cfg.llm.endpoints[cfg.llm.scenes.generate.primary]
    eligible = sorted((c for c in chunks if c['tokens'] > 0), key=lambda c: c['chunk_id'])
    random.Random(20261003).shuffle(eligible)
    selected = eligible[:size]
    data['token_requested'] = size
    data['token_available'] = len(eligible)
    data.setdefault('token_samples', [])
    successful = {r['chunk_id'] for r in data['token_samples']}
    sem = asyncio.Semaphore(min(concurrency, ep.max_concurrency))
    async with httpx.AsyncClient(timeout=cfg.llm.scenes.eval.timeout_s) as client:
        async def request(text):
            response = await client.post(ep.base_url.rstrip('/') + '/chat/completions',
                headers={'Authorization': 'Bearer ' + os.environ[ep.api_key_env]},
                json={**ep.extra_body, 'model': ep.model, 'messages': [
                    {'role': 'system', 'content': '阅读用户提供的资料。'},
                    {'role': 'user', 'content': text}], 'max_tokens': 1, 'temperature': 0})
            response.raise_for_status()
            value = response.json().get('usage', {}).get('prompt_tokens')
            if type(value) is not int or value < 1:
                raise ValueError('real prompt_tokens missing')
            return value
        if 'token_fixed_overhead' not in data:
            data['token_fixed_overhead'] = await request('')
            persist()
        overhead = data['token_fixed_overhead']
        async def one(chunk):
            if chunk['chunk_id'] in successful:
                return
            async with sem:
                try:
                    prompt = await request(chunk['text'])
                    if prompt <= overhead:
                        raise ValueError('chunk prompt does not exceed fixed overhead')
                    data['token_samples'].append({'chunk_id': chunk['chunk_id'], 'tokens': chunk['tokens'],
                        'prompt_tokens': prompt, 'fixed_overhead': overhead, 'model': ep.model})
                except Exception as exc:
                    data['failures'].append({'phase': 'tokens', 'chunk_id': chunk['chunk_id'],
                                             'error_type': type(exc).__name__})
                persist()
                print(f"token_samples={len(data['token_samples'])} actual_calls={current_summary().get('actual_calls', 0)}", flush=True)
        await asyncio.gather(*(one(c) for c in selected))


def answerable_cases(dataset, chunks, limit):
    # 包含keyword和不同类型；过时gold不能进入校准排名分母。
    cases = [c for c in dataset if c['expect']['answerable'] and c['gold'] and align_gold(c['gold'], chunks)]
    cases.sort(key=lambda c: ('keyword' not in c.get('tags', []), c['id']))
    return cases[:limit]


async def retrieve_observations(cfg, dataset, chunks, data, persist, *, limit):
    from ragagent.retrieval.filters import build_expr
    from ragagent.retrieval.hybrid import hybrid_one
    from ragagent.retrieval.merge import fetch_candidates
    from ragagent.retrieval.rerank import select
    chosen = answerable_cases(dataset, chunks, limit)
    negatives = [c for c in dataset if c['type'] == 'unanswerable'][:limit]
    data['retrieval_sample_ids'] = [c['id'] for c in chosen]
    embed = LocalEmbeddings.from_config(cfg)
    async with open_deps(cfg, cfg.services.milvus.collection) as deps, httpx.AsyncClient(timeout=60) as client:
        for case in chosen + negatives:
            question = case['expect'].get('standalone') or case['turns'][-1]['question']
            try:
                qvec = await embed.aembed_query(question)
                expr = build_expr({'doc_ids': case['expect'].get('scope_doc_ids', [])})
                hits = await asyncio.to_thread(hybrid_one, deps.milvus, cfg.services.milvus.collection,
                                              question, qvec, expr, cfg.retrieval)
                candidates = await asyncio.to_thread(fetch_candidates, deps.milvus, cfg.services.milvus.collection,
                                                     hits[:cfg.retrieval.candidates], cfg.retrieval)
                response = await client.post(embed.base_url + '/rerank', json={
                    'query': question, 'passages': [h.embed_text for h in candidates], 'max_length': cfg.rerank.max_length})
                response.raise_for_status(); scores = response.json()['scores']
                if len(scores) != len(candidates) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
                    raise ValueError('reranker scores invalid')
                data['grade_samples'].append({'id': case['id'], 'score': max(scores, default=0.),
                                              'label': bool(case['expect']['answerable'])})
                if case not in chosen:
                    persist(); continue
                gold = [sorted(g) for g in align_gold(case['gold'], chunks)]
                for threshold in [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]:
                    data['rerank_trials'].append({'id': case['id'], 'threshold': threshold, 'gold': gold,
                        'candidates': [{'chunk_id': h.chunk_id, 'score': s} for h, s in zip(candidates, scores)]})
                for weight in WEIGHTS:
                    rcfg = cfg.retrieval.model_copy(update={'weights': cfg.retrieval.weights.model_copy(
                        update={'dense': weight, 'sparse': round(1-weight, 2)})})
                    ranked = await asyncio.to_thread(hybrid_one, deps.milvus, cfg.services.milvus.collection,
                                                     question, qvec, expr, rcfg)
                    data['weight_trials'].append({'id': case['id'], 'dense_weight': weight,
                        'ranked': [h['chunk_id'] for h in ranked], 'gold': gold, 'tags': case.get('tags', [])})
                for count in [15, 30, 50]:
                    for search in [50, 100, 200]:
                        rcfg = cfg.retrieval.model_copy(update={'candidates': count, 'search_list': search})
                        start = time.perf_counter()
                        ranked = await asyncio.to_thread(hybrid_one, deps.milvus, cfg.services.milvus.collection,
                                                         question, qvec, expr, rcfg)
                        full = await asyncio.to_thread(fetch_candidates, deps.milvus, cfg.services.milvus.collection,
                                                       ranked[:count], rcfg)
                        res = await client.post(embed.base_url + '/rerank', json={'query': question,
                            'passages': [h.embed_text for h in full], 'max_length': cfg.rerank.max_length})
                        res.raise_for_status()
                        selected = select([h.model_dump() for h in full], res.json()['scores'], cfg.rerank)
                        data['retrieval_trials'].append({'id': case['id'], 'candidates': count, 'search_list': search,
                            'latency_s': time.perf_counter()-start, 'ranked': [h['chunk_id'] for h in selected], 'gold': gold})
                persist()
                print(f"retrieval_sample={case['id']} weights=7 operating_points=9", flush=True)
            except Exception as exc:
                data['failures'].append({'phase': 'retrieval', 'id': case['id'], 'error_type': type(exc).__name__})
                persist()


async def chunk_observations(cfg, dataset, chunks, data, persist, *, limit, prefix):
    from ragagent.ingest.models import CleanDocument
    from ragagent.ingest.chunking.pipeline import run_chunk
    from ragagent.ingest.chunking.tokenizer import get_counter
    from ragagent.index.embedder import Embedder
    from ragagent.index.milvus import ensure_collection, wait_indexed, load
    from ragagent.index.writer import index_document, flush
    from ragagent.retrieval.hybrid import hybrid_one
    from ragagent.retrieval.filters import build_expr
    from ragagent.retrieval.merge import fetch_candidates
    from ragagent.retrieval.rerank import select
    if not prefix.startswith('l4_') or not prefix.replace('_', '').isalnum():
        raise ValueError('temporary collection must have safe l4_ prefix')
    cases = answerable_cases(dataset, chunks, limit)
    required = {g['doc_id'] for case in cases for g in case['gold']}
    others = sorted({c['doc_id'] for c in chunks} - required)
    random.Random(20261003).shuffle(others)
    noise = others[:math.ceil(len(others)*.05)]
    docs = []
    for doc_id in sorted(required | set(noise)):
        paths = sorted((Path(cfg.paths.clean)/doc_id).glob('v*/clean.json'))
        if not paths:
            raise ValueError('clean document missing')
        latest = max(paths, key=lambda p: int(p.parent.name[1:]))
        docs.append(CleanDocument.model_validate_json(latest.read_text()))
    data['chunk_experiment'] = {'evaluation_samples': [c['id'] for c in cases],
        'gold_documents': sorted(required), 'noise_documents': noise, 'targets': list(CHUNK_TARGETS)}
    count = get_counter(cfg.chunking.tokenizer)
    embed = LocalEmbeddings.from_config(cfg)
    async with open_deps(cfg, cfg.services.milvus.collection) as deps, httpx.AsyncClient(timeout=60) as http:
        with Embedder(cfg=cfg) as encoder:
            for target in CHUNK_TARGETS:
                collection = f'{prefix}_{target}'
                try:
                    # 只新建本轮名称，存在时停止，避免污染其他实验。
                    if deps.milvus.has_collection(collection):
                        raise ValueError('temporary collection already exists')
                    ensure_collection(deps.milvus, collection, cfg.index.dim)
                    experiment_chunks = []
                    for doc in docs:
                        ccfg = cfg.chunking.model_copy(update={'target_tokens': target})
                        parts = await asyncio.to_thread(run_chunk, doc, cfg=ccfg, count=count, encode=encoder.text)
                        await asyncio.to_thread(index_document, deps.milvus, collection, encoder, doc, parts, cfg=cfg.index)
                        experiment_chunks.extend(c.model_dump() for c in parts)
                        print(f'chunk_target={target} document={doc.doc_id} chunks={len(parts)}', flush=True)
                    await asyncio.to_thread(flush, deps.milvus, collection)
                    indexes = await asyncio.to_thread(wait_indexed, deps.milvus, collection)
                    await asyncio.to_thread(load, deps.milvus, collection)
                    data.setdefault('temporary_collections', []).append({'collection': collection,
                        'target_tokens': target, 'chunks': len(experiment_chunks), 'documents': len(docs),
                        'index_progress': indexes})
                    for case in cases:
                        gold = align_gold(case['gold'], experiment_chunks)
                        if gold is None:
                            data['failures'].append({'phase': 'chunks', 'target_tokens': target,
                                                     'id': case['id'], 'error_type': 'StaleGold'})
                            continue
                        question = case['expect'].get('standalone') or case['turns'][-1]['question']
                        vector = await embed.aembed_query(question)
                        expr = build_expr({'doc_ids': case['expect'].get('scope_doc_ids', [])})
                        start = time.perf_counter()
                        hits = await asyncio.to_thread(hybrid_one, deps.milvus, collection, question, vector, expr, cfg.retrieval)
                        full = await asyncio.to_thread(fetch_candidates, deps.milvus, collection,
                                                       hits[:cfg.retrieval.candidates], cfg.retrieval)
                        response = await http.post(embed.base_url + '/rerank', json={'query': question,
                            'passages': [h.embed_text for h in full], 'max_length': cfg.rerank.max_length})
                        response.raise_for_status()
                        selected = select([h.model_dump() for h in full], response.json()['scores'], cfg.rerank)
                        data['chunk_trials'].append({'id': case['id'], 'target_tokens': target,
                            'ranked': [h['chunk_id'] for h in selected], 'gold': [sorted(g) for g in gold],
                            'latency_s': time.perf_counter()-start})
                    persist()
                except Exception as exc:
                    data['failures'].append({'phase': 'chunks', 'target_tokens': target, 'error_type': type(exc).__name__})
                    persist()


class SupportLabel(BaseModel):
    supported: bool


async def support_observations(cfg, artifacts, chunks, data, persist, *, limit):
    """完整档句子引用用独立评测模型标注；沿用生产解析和token窗口。"""
    from ragagent.llm.structured import structured_call
    from ragagent.postprocess.verify import windows
    from ragagent.postprocess.parse import mask_code, find_supplement, split_sentences
    from ragagent.context.models import ContextBlock
    from ragagent.ingest.chunking.tokenizer import get_counter
    count = get_counter(cfg.chunking.tokenizer)
    texts = {c['chunk_id']: c['text'] for c in chunks}
    ep = LocalEmbeddings.from_config(cfg)
    async with httpx.AsyncClient(timeout=60) as client:
        examined = 0
        for artifact in artifacts:
            state = artifact['state']; answer = state.get('answer') or ''
            if not isinstance(answer, str):
                continue
            blocks = {b['cite']: ContextBlock.model_validate(b) for b in state.get('context_blocks') or []}
            masked = mask_code(answer)
            sentences = split_sentences(answer, masked, find_supplement(masked, cfg.prompt.supplement_heading))
            for sentence in sentences:
                if sentence.in_supplement or len(sentence.claim) < cfg.postprocess.min_claim_chars:
                    continue
                for citation in sentence.cites:
                    if examined >= limit:
                        return
                    if citation['n'] not in blocks:
                        continue
                    examined += 1
                    try:
                        block = blocks[citation['n']]
                        text = sentence.claim
                        bound = max(1, cfg.postprocess.rerank_max_length - count(text) - 4)
                        passages = windows(block, texts, count, bound)
                        response = await client.post(ep.base_url + '/rerank', json={'query': text,
                            'passages': passages, 'max_length': cfg.postprocess.rerank_max_length})
                        response.raise_for_status(); scores = response.json()['scores']
                        if len(scores) != len(passages) or not scores or any(not 0 <= x <= 1 for x in scores):
                            raise ValueError('support rerank scores invalid')
                        score = max(scores)
                        label, endpoint = await structured_call(cfg.llm.scenes.eval, cfg.llm.endpoints, SupportLabel,
                            [SystemMessage(content='判断资料是否支持句子中的全部事实，不能用外部知识，返回supported布尔。'),
                             HumanMessage(content=json.dumps({'sentence': text, 'reference': passages}, ensure_ascii=False))])
                        if label is None:
                            raise ValueError('support label missing')
                        data['support_pairs'].append({'id': artifact['case']['id'], 'cite': citation['n'],
                            'score': score, 'label': SupportLabel.model_validate(label).supported, 'endpoint': endpoint})
                    except Exception as exc:
                        data['failures'].append({'phase': 'support', 'id': artifact['case']['id'], 'error_type': type(exc).__name__})
                    persist()
                    print(f"support_pairs={len(data['support_pairs'])} actual_calls={current_summary().get('actual_calls', 0)}", flush=True)


async def run(args):
    cfg = load_config(args.config)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(output.read_text()) if output.exists() else {
        'provenance': {'mode': 'real', 'collection': cfg.services.milvus.collection,
            'config_hash': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            'chunks_hash': hashlib.sha256(Path(args.chunks).read_bytes()).hexdigest(),
            'seed': 20261003, 'no_langfuse': True}, 'failures': [], 'grade_samples': [], 'rerank_trials': [],
        'weight_trials': [], 'retrieval_trials': [], 'chunk_trials': [], 'support_pairs': []}
    chunks = load_rows(args.chunks)
    with metrics_scope(None), counted_calls(args.max_calls, ledger=output.with_suffix(f'.{args.phase}.calls.json')):
        def persist():
            data.setdefault('phase_calls', {})[args.phase] = current_summary()
            save(output, data)
        try:
            if args.phase == 'tokens':
                await token_samples(cfg, chunks, data, persist, size=args.token_samples, concurrency=args.concurrency)
            elif args.phase == 'retrieval':
                await retrieve_observations(cfg, load_rows(args.dataset), chunks, data, persist, limit=args.samples)
            elif args.phase == 'chunks':
                await chunk_observations(cfg, load_rows(args.dataset), chunks, data, persist,
                                         limit=args.samples, prefix=args.collection_prefix)
            elif args.phase == 'support':
                await support_observations(cfg, load_rows(args.states), chunks, data, persist, limit=args.samples)
        except Exception as exc:
            data['failures'].append({'phase': args.phase, 'error_type': type(exc).__name__})
        finally:
            from ragagent.llm.client import close_clients
            await close_clients()
            persist()
    print(json.dumps({'phase': args.phase, 'actual_calls': data['phase_calls'][args.phase]['actual_calls'],
                      'failures': len(data['failures'])}))
    return 1 if any(f['phase'] == args.phase for f in data['failures']) else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['tokens', 'retrieval', 'chunks', 'support'], required=True)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--chunks', required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--states')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-calls', type=int, required=True)
    parser.add_argument('--samples', type=int, default=60)
    parser.add_argument('--token-samples', type=int, default=1000)
    parser.add_argument('--concurrency', type=int, default=6)
    parser.add_argument('--collection-prefix', default='l4_calibration_20261003')
    args = parser.parse_args(argv)
    if args.samples < 1 or args.token_samples < 1 or args.concurrency < 1:
        parser.error('sample and concurrency limits must be positive')
    if args.phase in {'retrieval', 'chunks'} and not args.dataset:
        parser.error('--dataset is required for retrieval/chunks')
    if args.phase == 'support' and not args.states:
        parser.error('--states is required for support')
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
