"""§8 递归拆分：精确计数、无损片段、结构重复仅在展示层。

restore_ranges 是输出 text 中属于原始 Markdown 的字符区间；source_spans
是它们对应的输入区间。生成围栏、重复表头、overlap 不参与原文恢复。
"""
from dataclasses import dataclass, field
import re

from .models import Unit
from .units import code_parts, render_block, unit_text

SEPARATORS = [r'(?<=\n\n)', r'(?<=\n)', r'(?<=[。！？；!?;])', r'(?<=[，,、])']


@dataclass
class Fragment:
    text: str
    blocks: list
    kind: str = 'text'
    meta: dict = field(default_factory=dict)
    source_spans: list = field(default_factory=list)
    restore_ranges: list = field(default_factory=list)


def hard_cut(text: str, target: int, count) -> list[str]:
    """按原始字符边界切，绝不 decode token 导致 U+FFFD 或空白丢失。"""
    out = []
    while text:
        if count(text) <= target:
            out.append(text)
            break
        lo, hi, best = 1, len(text), 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if count(text[:mid]) <= target:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        if not best:
            raise ValueError('预算不足以容纳一个 Unicode 字符')
        out.append(text[:best])
        text = text[best:]
    return out


def split_text(text: str, target: int, count, level: int = 0) -> list[str]:
    if count(text) <= target:
        return [text]
    if level == len(SEPARATORS):
        return hard_cut(text, target, count)
    return [part for p in re.split(SEPARATORS[level], text) if p
            for part in split_text(p, target, count, level + 1)]


def repack(pieces, target, overlap, count):
    parts, buf, head = [], [], 0
    for p in pieces:
        if buf and count(''.join(buf) + p) > target:
            parts.append((''.join(buf), head))
            tail = []
            for q in reversed(buf):
                if count(q + ''.join(tail)) > overlap:
                    break
                tail.insert(0, q)
            if len(tail) == len(buf):
                tail = []
            while tail and count(''.join(tail) + p) > target:
                tail.pop(0)
            head = len(''.join(tail))
            buf = tail
        buf.append(p)
    if buf:
        parts.append((''.join(buf), head))
    return parts


def whole(unit):
    text, spans, ranges, pos = '', [], [], 0
    for b in unit.blocks:
        s = render_block(b)
        text += s
        if b.type != 'image':
            spans.append(b.meta['_span'])
            ranges.append((pos, pos + len(s)))
        pos += len(s)
    meta = {}
    if unit.kind == 'code':
        body, lang, _ = code_parts(unit.blocks[0])
        meta = {'lang':lang, 'code_body_span':(0,len(body)), 'force_split':False}
    return Fragment(text=text, blocks=unit.blocks, kind=unit.kind, meta=meta,
                    source_spans=spans, restore_ranges=ranges)


def _text_fragments(unit, cfg, count):
    text = unit_text(unit)
    parts = repack(split_text(text, cfg.target_tokens, count), cfg.target_tokens,
                   cfg.overlap_tokens, count)
    pos, base = 0, unit.blocks[0].meta['_span'][0]
    result = []
    # 通常超长 Unit 为单块；绑定超过 max 已解除。图片占位符不算原文。
    for part, head in parts:
        size = len(part) - head
        if unit.blocks[0].type == 'image':
            spans, ranges = [], []
        else:
            spans, ranges = [(base + pos, base + pos + size)], [(head, len(part))]
        result.append(Fragment(text=part, blocks=unit.blocks, kind=unit.kind,
                               meta={'overlap_chars': head}, source_spans=spans,
                               restore_ranges=ranges))
        pos += size
    return result


def _code_fragments(unit, cfg, count):
    b = unit.blocks[0]
    body, lang, fence = code_parts(b)
    prefix, suffix = f'{fence}{lang}\n', f'{fence}\n\n'
    if count(prefix + '\n' + suffix) >= cfg.max_tokens:
        raise ValueError('max_tokens 不足以容纳代码围栏及语言标签')
    # 优先顶层定义，再空行，再行；只有无法容纳的一行才做语法/字符拆分。
    patterns = {'python': r'(?=^(?:def |class |async def ))',
                'javascript': r'(?=^(?:function |class |export ))',
                'typescript': r'(?=^(?:function |class |export ))',
                'go': r'(?=^(?:func |type ))'}
    pattern = patterns.get(lang)
    coarse = re.split(pattern, body, flags=re.M) if pattern else [body]
    pieces = []
    for item in coarse:
        if not item:
            continue
        if count(prefix + item + ('' if item.endswith('\n') else '\n') + suffix) <= cfg.max_tokens:
            pieces.append((item, False))
            continue
        for paragraph in re.split(r'(?<=\n\n)', item):
            if not paragraph:
                continue
            if count(prefix + paragraph + ('' if paragraph.endswith('\n') else '\n') + suffix) <= cfg.max_tokens:
                pieces.append((paragraph, False))
                continue
            for line in paragraph.splitlines(keepends=True):
                if count(prefix + line + ('' if line.endswith('\n') else '\n') + suffix) <= cfg.max_tokens:
                    pieces.append((line, False))
                else:
                    # 使用精确完整输出计数，给围栏留预算。
                    budget = max(1, min(cfg.target_tokens, cfg.max_tokens - count(prefix + '\n' + suffix) - 4))
                    segments = re.split(r'(?<=[,;{}()])', line)
                    for segment in segments:
                        for part in hard_cut(segment, budget, count):
                            # 非加性 tokenizer 下进一步缩小，避免合并边界误差。
                            while count(prefix + part + ('' if part.endswith('\n') else '\n') + suffix) > cfg.max_tokens:
                                budget -= 1
                                if budget <= 0:
                                    raise ValueError('代码预算无法容纳字符')
                                # 由后续预算安全分拆，而非截断内容。
                                break
                            if count(prefix + part + ('' if part.endswith('\n') else '\n') + suffix) > cfg.max_tokens:
                                pieces.extend((q, True) for q in hard_cut(part, budget, count))
                            else:
                                pieces.append((part, True))
    groups, buf, forced = [], '', False
    for p, force in pieces:
        candidate = buf + p
        limit = max(cfg.target_tokens, count(prefix + p + ('' if p.endswith('\n') else '\n') + suffix))
        if buf and count(prefix + candidate + ('' if candidate.endswith('\n') else '\n') + suffix) > min(limit, cfg.max_tokens):
            groups.append((buf, forced))
            buf, forced = '', False
        buf += p
        forced |= force
    if buf:
        groups.append((buf, forced))
    result, pos = [], 0
    base = b.meta['_span'][0]
    for i, (payload, force) in enumerate(groups):
        added_newline = '' if payload.endswith('\n') else '\n'
        text = prefix + payload + added_newline + suffix
        left = 0 if i == 0 else len(prefix)
        right = len(text) if i == len(groups)-1 else len(prefix) + len(payload)
        # 原始最后一段已有换行；中间新增换行/围栏不参与恢复。
        start = base if i == 0 else base + len(prefix) + pos
        end = base + len(render_block(b)) if i == len(groups)-1 else base + len(prefix) + pos + len(payload)
        result.append(Fragment(text=text, blocks=[b], kind='code',
            meta={'lang':lang, 'force_split':force, 'code_body_span':(pos,pos+len(payload))},
            source_spans=[(start,end)], restore_ranges=[(left,right)]))
        pos += len(payload)
    return result


def _row_fragments(unit, cfg, count):
    """表格重复表头，列表顶层项优先；超长行/项才递归兜底。"""
    b = unit.blocks[0]
    text = render_block(b)
    if unit.kind == 'table':
        lines = text.splitlines(keepends=True)
        divider = next((i for i,s in enumerate(lines) if re.match(r'^\s*\|?\s*:?-{3}',s)), None)
        if divider is None:
            return _text_fragments(unit,cfg,count)
        header = ''.join(lines[:divider+1])
        rows = lines[divider+1:]
        # 表头本身超过预算无法重复，退回无损文本兜底且显式标识。
        if count(header) >= cfg.max_tokens - 4:
            result = _text_fragments(unit,cfg,count)
            for f in result:
                f.meta['force_split'] = True
            return result
    else:
        header = ''
        rows = re.split(r'(?=^(?:\s*[-+*] |\d+[.)] ))',text,flags=re.M)
        rows = [r for r in rows if r]
    groups, buf = [], ''
    for row in rows:
        if count(header + row) > cfg.max_tokens:
            if buf:
                groups.append((buf, False)); buf = ''
            # 罕见超长单行表格/列表，重复表头并精确兜底。
            budget = max(1, min(cfg.target_tokens,count(row),cfg.max_tokens - count(header) - 8))
            for q in hard_cut(row,budget,count):
                groups.append((q, True))
            continue
        if buf and count(header + buf + row) > cfg.target_tokens:
            groups.append((buf, False)); buf = ''
        buf += row
    if buf:
        groups.append((buf, False))
    result, pos, base = [], len(header), b.meta['_span'][0]
    for i,(body,force) in enumerate(groups):
        part = header + body
        left = 0 if i == 0 else len(header)
        start = base if i == 0 else base+pos
        result.append(Fragment(text=part,blocks=[b],kind=unit.kind,
            meta={'repeated_header_chars': 0 if i == 0 else len(header),'force_split':force},
            source_spans=[(start,base+pos+len(body))],restore_ranges=[(left,len(part))]))
        pos += len(body)
    return result


def split_unit(unit: Unit, cfg, count) -> list[Fragment]:
    if count(unit_text(unit)) <= cfg.max_tokens:
        return [whole(unit)]
    if unit.kind == 'code':
        parts = _code_fragments(unit,cfg,count)
    elif unit.kind in ('table','list'):
        parts = _row_fragments(unit,cfg,count)
    else:
        parts = _text_fragments(unit,cfg,count)
    for i,f in enumerate(parts):
        f.meta.update(strategy='split',split_part=f'{i+1}/{len(parts)}')
        if count(f.text) > cfg.max_tokens:
            raise ValueError('结构拆分输出超过 max_tokens')
    return parts
