"""等长屏蔽 Markdown 代码；解析位置始终是 Python 原文下标。"""
from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field

# 输出仍要求 [n]；仅解析成对的兼容格式，不规范化原文或接受混配括号。
CITE = re.compile(r"(?:\[(\d{1,3})\]|【(\d{1,3})】)")
END = re.compile(r"[。！？!?；;]|\.(?=\s|$)|\n")
MD = re.compile(r"[*_#>`|]+|^\s*[-+]\s+|^\s*\d+\.\s+", re.M)
FENCE_START = re.compile(r"^[ \t]*(`{3,}|~{3,})")


class Sentence(BaseModel):
    start: int
    end: int
    claim: str
    cites: list[dict] = Field(default_factory=list)
    in_supplement: bool = False


def _code_spans(text: str) -> list[tuple[int, int, int]]:
    """代码范围及行内定界符长度；围栏的长度记为 0。"""
    spans, opened, pos = [], None, 0
    for line in text.splitlines(keepends=True):
        m = FENCE_START.match(line)
        if opened:
            if m and m[1][0] == opened[0] and len(m[1]) >= opened[1] and not line[m.end():].strip():
                spans.append((opened[2], pos + len(line), 0))
                opened = None
        elif m:
            opened = (m[1][0], len(m[1]), pos)
        pos += len(line)
    if opened:
        spans.append((opened[2], len(text), 0))
    regions, cursor = [], 0
    for start, stop, _ in spans:
        regions.append((cursor, start))
        cursor = stop
    regions.append((cursor, len(text)))
    for left, right in regions:
        runs = list(re.finditer(r'`+', text[left:right]))
        index = 0
        while index < len(runs):
            first = runs[index]
            closing = next((j for j in range(index + 1, len(runs))
                            if len(runs[j][0]) == len(first[0])), None)
            if closing is None:
                index += 1
            else:
                spans.append((left + first.start(), left + runs[closing].end(), len(first[0])))
                index = closing + 1
    return sorted(spans)


def mask_code(text: str) -> str:
    chars = list(text)
    for start, stop, _ in _code_spans(text):
        chars[start:stop] = [' ' if c not in '\r\n' else c for c in chars[start:stop]]
    return ''.join(chars)


def _claim_text(answer: str, start: int, stop: int, code_spans, cites) -> str:
    # 仅删除已识别的引用；行内代码的下标、下划线和运算符保留原意。
    spans = sorted([*code_spans, *((c['start'], c['end'], -1) for c in cites)])
    parts, pos = [], start
    for left, right, ticks in spans:
        if right <= start or left >= stop:
            continue
        parts.append(MD.sub('', answer[pos:max(pos, left)]))
        if ticks > 0:
            parts.append(answer[max(start, left + ticks):min(stop, right - ticks)])
        elif ticks == 0:
            parts.append(' ')
        pos = min(stop, right)
    parts.append(MD.sub('', answer[pos:stop]))
    return ''.join(parts).strip()


def _norm(text: str) -> str:
    return re.sub(r'[\s*_#]', '', unicodedata.normalize('NFKC', text))


def find_supplement(masked: str, heading: str) -> int | None:
    target, pos = _norm(heading), 0
    for line in masked.splitlines(keepends=True):
        if _norm(line) == target:
            return pos
        pos += len(line)
    return None


def split_sentences(answer: str, masked: str, supp_at: int | None) -> list[Sentence]:
    out, start = [], 0
    code_spans = _code_spans(answer)
    while start < len(masked):
        m = END.search(masked, start)
        stop = m.end() if m else len(masked)
        while (cite := CITE.match(masked, stop)):
            stop = cite.end()
        seg = masked[start:stop]
        cites = [{'n': int(c[1] or c[2]), 'start': start + c.start(), 'end': start + c.end()}
                 for c in CITE.finditer(seg)]
        claim = _claim_text(answer, start, stop, code_spans, cites)
        if claim or cites:
            out.append(Sentence(start=start, end=stop, claim=claim, cites=cites,
                                in_supplement=supp_at is not None and start >= supp_at))
        start = stop
    return out


def check_format(sents: list[Sentence], blocks: dict) -> None:
    for sent in sents:
        for cite in sent.cites:
            cite['state'] = ('in_supplement' if sent.in_supplement else
                             'invalid_number' if cite['n'] not in blocks else 'unverified')


def citation_format_issues(masked: str, blocks: dict) -> list[dict]:
    """记录已知引用语法违规；不猜测来源，不把普通括号文字当成引用。"""
    titles = {b.title.rstrip('：:').strip() for b in blocks.values()}
    issues = []
    for match in re.finditer(r'【([^【】\n]+)】|cite[^\n]*', masked):
        token, body = match[0], match[1]
        if body is None:
            kind = 'foreign_marker'
        elif re.match(r'^\d{1,3}\s*[:：]', body):
            kind = 'number_with_title'
        elif body.rstrip('：:').strip() in titles:
            kind = 'title_without_number'
        else:
            continue
        issues.append(dict(kind=kind, text=token, start=match.start(), end=match.end()))
    return issues
