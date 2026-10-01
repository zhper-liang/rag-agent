"""完整问答轮次预算；旧历史只去掉代码外的引用。"""
from __future__ import annotations

from ragagent.postprocess.parse import CITE, mask_code


def strip_old_citations(text: str) -> str:
    spans = [(m.start(), m.end()) for m in CITE.finditer(mask_code(text))]
    pos, out = 0, []
    for start, end in spans:
        out.append(text[pos:start])
        pos = end
    return ''.join(out) + text[pos:]


def truncate_tokens(text: str, limit: int, count, suffix='……（已截断）') -> str:
    if count(text) <= limit:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(text[:mid] + suffix) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + suffix


def history_turns(messages) -> list[tuple[str, str]]:
    out, question = [], None
    for msg in messages:
        if isinstance(msg, dict):
            role, text, extra = msg.get('role', msg.get('type')), msg.get('content'), msg.get('additional_kwargs', {})
        else:
            role, text, extra = msg.type, msg.content, msg.additional_kwargs
        if not isinstance(text, str):
            continue  # 不重新发送历史图片
        if role in ('human', 'user'):
            question = text
        elif role in ('ai', 'assistant') and question is not None:
            plain = extra.get('plain')
            out.append((question, plain if isinstance(plain, str) else strip_old_citations(text)))
            question = None
    return out


def trim_history(turns, cfg, count) -> list[dict]:
    kept, total = [], 0
    for question, answer in reversed(turns[-cfg.history_turns:]):
        answer = truncate_tokens(answer, cfg.history_answer_tokens, count)
        cost = count(question) + count(answer)
        if total + cost > cfg.history_total_tokens:
            break
        kept.insert(0, (question, answer))
        total += cost
    return [dict(role=role, content=text) for q, a in kept
            for role, text in [('user', q), ('assistant', a)]]
