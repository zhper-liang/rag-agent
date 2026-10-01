"""引用-窗口批量打分；验证整个响应后再原子地修改引用状态。"""
from __future__ import annotations

import asyncio
import math


class VerifyDegraded(Exception):
    pass


def token_windows(text: str, limit: int, count) -> list[str]:
    if not text:
        return []
    out, start = [], 0
    while start < len(text):
        if count(text[start:]) <= limit:
            out.append(text[start:])
            break
        lo, hi = 0, len(text) - start
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count(text[start:start + mid]) <= limit:
                lo = mid
            else:
                hi = mid - 1
        # 不能切开字符；即使计数器认为一个字符就超限也必须前进。
        stop = start + max(lo, 1)
        out.append(text[start:stop])
        start = stop
    return out


def windows(block, chunk_texts: dict, count, limit: int) -> list[str]:
    texts = [chunk_texts[cid] for cid in block.chunk_ids if chunk_texts.get(cid)]
    return [w for text in (texts or [block.text]) for w in token_windows(text, limit, count)]


async def check_support(sents, blocks, chunk_texts, rerank_pairs, cfg, *, count=len) -> None:
    jobs = []
    for sent in sents:
        if sent.in_supplement or len(sent.claim) < cfg.min_claim_chars:
            continue
        for cite in sent.cites:
            if cite['state'] != 'unverified':
                continue
            limit = max(1, cfg.rerank_max_length - count(sent.claim) - 4)
            for text in windows(blocks[cite['n']], chunk_texts, count, limit):
                jobs.append((cite, sent.claim, text))
                if len(jobs) >= cfg.max_pairs:
                    break
            if len(jobs) >= cfg.max_pairs:
                break
        if len(jobs) >= cfg.max_pairs:
            break
    if not jobs:
        return
    try:
        raw = await asyncio.wait_for(rerank_pairs([(q, w) for _, q, w in jobs]), cfg.verify_timeout_s)
        if not isinstance(raw, (tuple, list)) or len(raw) != len(jobs):
            raise ValueError('invalid scores length')
        scores = []
        for score in raw:
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('invalid score')
            scores.append(float(score))
    except Exception as exc:
        # 不携带 HTTP/模型异常正文，避免泄漏资料。
        raise VerifyDegraded(type(exc).__name__) from None
    best = {}
    for (cite, _, _), score in zip(jobs, scores, strict=True):
        best[id(cite)] = max(best.get(id(cite), 0.), score)
    for cite, _, _ in jobs:
        cite['score'] = best[id(cite)]
        cite['state'] = 'ok' if cite['score'] >= cfg.support_threshold else 'unsupported'


def support_warning(sents, cfg) -> bool:
    checked = [s for s in sents if not s.in_supplement and
               any(c['state'] in ('ok', 'unsupported') for c in s.cites)]
    unsupported = sum(not any(c['state'] == 'ok' for c in s.cites) for s in checked)
    return bool(checked and unsupported / len(checked) > cfg.warn_ratio)
