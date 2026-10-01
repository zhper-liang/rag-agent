"""只附加 final，不改已推送的 answer；同步生成可存入历史的 plain。"""
from __future__ import annotations

import time
from collections import Counter
from uuid import uuid4

from ragagent.context.models import ContextBlock
from ragagent.postprocess.offsets import utf16_citations, utf16_offset
from ragagent.postprocess.parse import check_format, find_supplement, mask_code, split_sentences
from ragagent.postprocess.plain import make_plain
from ragagent.postprocess.sources import build_sources
from ragagent.postprocess.verify import VerifyDegraded, check_support, support_warning


async def build_final(state, cfg, rerank_pairs, *, count=len, message_id=None):
    start = time.perf_counter()
    answer, meta = state.get('answer', ''), state.get('gen_meta', {})
    status = meta.get('status', 'ok')
    if status == 'failed':
        return None, None, {}
    mode = meta.get('scene', state.get('intent', 'rag'))
    sents, supp_at, degraded = [], None, False
    blocks = {b.cite: b for b in (ContextBlock.model_validate(b) for b in state.get('context_blocks', []))}
    if mode in ('rag', 'rag_empty'):
        masked = mask_code(answer)
        supp_at = find_supplement(masked, cfg.prompt.supplement_heading)
        sents = split_sentences(answer, masked, supp_at)
        check_format(sents, blocks)
        chunks = list(state.get('hits', [])) + list(state.get('candidates', []))
        chunks += [c for cs in state.get('expansions', {}).values() for c in cs]
        chunk_texts = {c['chunk_id']: c['text'] for c in chunks if 'chunk_id' in c and 'text' in c}
        try:
            await check_support(sents, blocks, chunk_texts, rerank_pairs, cfg.postprocess, count=count)
        except VerifyDegraded:
            degraded = True
    warned = support_warning(sents, cfg.postprocess)
    notices = [cfg.postprocess.notices.unsupported] if warned else []
    if status in ('truncated', 'interrupted'):
        notices.append(getattr(cfg.postprocess.notices, status))
    citations = utf16_citations(answer, sents)
    event = dict(type='final', message_id=message_id or uuid4().hex, status=status,
                 notice='\n'.join(notices) or None,
                 supplement={'start': utf16_offset(answer, supp_at)} if supp_at is not None else None,
                 citations=citations, sources=build_sources(sents, blocks))
    body = [s for s in sents if s.claim and not s.in_supplement]
    covered = sum(bool(s.cites) for s in body)
    stats = dict(citations=len(citations), states=dict(Counter(c['state'] for c in citations)),
                 coverage=covered / len(body) if body else 0., covered=covered,
                 uncovered=len(body) - covered, warned=warned, verify_degraded=degraded,
                 verify_ms=round((time.perf_counter() - start) * 1000))
    return event, make_plain(answer, sents, status, cfg.postprocess), stats
