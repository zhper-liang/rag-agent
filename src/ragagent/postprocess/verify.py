"""引用-窗口批量打分；验证整个响应后再原子地修改引用状态。"""
from __future__ import annotations

import asyncio
import math
import re
from itertools import zip_longest


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


def claim_query(claim: str, question: str) -> str:
    """为省略函数名的回答补上问题中的代码实体，不把问题当作事实证据。"""
    identifiers = list(dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", question)))
    missing = [word for word in identifiers if word not in claim]
    return f"{'、'.join(missing)}：{claim}" if missing else claim


async def check_support(sents, blocks, chunk_texts, rerank_pairs, cfg, *, count=len, question='') -> None:
    groups = []
    for sent in sents:
        if sent.in_supplement or len(sent.claim) < cfg.min_claim_chars:
            continue
        query = claim_query(sent.claim, question)
        for cite in sent.cites:
            if cite['state'] != 'unverified':
                continue
            limit = max(1, cfg.rerank_max_length - count(query) - 4)
            texts = windows(blocks[cite['n']], chunk_texts, count, limit)
            groups.append([(cite, query, text) for text in texts])
    # 每条引用先获得一个窗口，再轮流分配后续窗口，避免开头引用耗尽预算。
    jobs = [job for level in zip_longest(*groups) for job in level if job is not None][:cfg.max_pairs]
    totals = {id(group[0][0]): len(group) for group in groups if group}
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
    best, checked = {}, {}
    for (cite, _, _), score in zip(jobs, scores, strict=True):
        best[id(cite)] = max(best.get(id(cite), 0.), score)
        checked[id(cite)] = checked.get(id(cite), 0) + 1
    for cite, _, _ in jobs:
        cite['score'] = best[id(cite)]
        cite['state'] = ('ok' if cite['score'] >= cfg.support_threshold else
                         'unsupported' if checked[id(cite)] == totals[id(cite)] else 'unverified')


def support_warning(sents, cfg) -> bool:
    checked = [s for s in sents if not s.in_supplement and
               any(c['state'] in ('ok', 'unsupported') for c in s.cites)]
    unsupported = sum(not any(c['state'] == 'ok' for c in s.cites) for s in checked)
    return bool(checked and unsupported / len(checked) > cfg.warn_ratio)
