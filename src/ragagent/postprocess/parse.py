"""等长屏蔽 Markdown 代码；解析位置始终是 Python 原文下标。"""
from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field

CITE = re.compile(r"\[(\d{1,3})\]")
END = re.compile(r"[。！？!?；;]|\.(?=\s|$)|\n")
MD = re.compile(r"[*_#>`|]+|^\s*[-+]\s+|^\s*\d+\.\s+", re.M)
FENCE_START = re.compile(r"^[ \t]*(`{3,}|~{3,})")


class Sentence(BaseModel):
    start: int
    end: int
    claim: str
    cites: list[dict] = Field(default_factory=list)
    in_supplement: bool = False


def mask_code(text: str) -> str:
    chars = list(text)

    def blank(a, b):
        for i in range(a, b):
            if chars[i] not in '\r\n':
                chars[i] = ' '

    opened, pos = None, 0
    for line in text.splitlines(keepends=True):
        m = FENCE_START.match(line)
        if opened:
            blank(pos, pos + len(line))
            if m and m[1][0] == opened[0] and len(m[1]) >= opened[1] and not line[m.end():].strip():
                opened = None
        elif m:
            opened = (m[1][0], len(m[1]))
            blank(pos, pos + len(line))
        pos += len(line)
    # 单/多反引号行内代码；找同长度定界符，不把更长的一段当成闭合。
    masked = ''.join(chars)
    runs = list(re.finditer(r'`+', masked))
    index = 0
    while index < len(runs):
        first = runs[index]
        closing = next((j for j in range(index + 1, len(runs))
                        if len(runs[j][0]) == len(first[0])), None)
        if closing is None:
            index += 1
        else:
            blank(first.start(), runs[closing].end())
            index = closing + 1
    return ''.join(chars)


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
    while start < len(masked):
        m = END.search(masked, start)
        stop = m.end() if m else len(masked)
        while (cite := CITE.match(masked, stop)):
            stop = cite.end()
        seg = masked[start:stop]
        cites = [{'n': int(c[1]), 'start': start + c.start(), 'end': start + c.end()}
                 for c in CITE.finditer(seg)]
        claim = MD.sub('', CITE.sub('', seg)).strip()
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
