"""每条规则均独立、非原地修改的 list[Block]→list[Block]。"""
from collections import defaultdict
import re
import unicodedata

from markdown_it import MarkdownIt

from ragagent.config import load_config
from ragagent.ingest.models import Block


def _cfg(cfg):
    return cfg or load_config().ingest.clean


def normalize(blocks: list[Block]) -> list[Block]:
    result = []
    for block in blocks:
        b = block.model_copy(deep=True)
        # 代码与LaTeX逐字保留；表格保留换行/空列/围栏，不合并结构空白。
        if b.type not in {'code', 'equation'}:
            value = unicodedata.normalize('NFKC', b.text)
            value = ''.join(c for c in value if unicodedata.category(c) not in {'Cf', 'Cc'} or c in '\n\t')
            if b.type not in {'table', 'list'}:
                value = re.sub(r'\s+', ' ', value)
            else:
                value = '\n'.join(re.sub(r'[ \t]+', ' ', line).strip() for line in value.splitlines()) if b.type == 'table' else value.rstrip()
            value = re.sub(r'(?<=[\u3400-\u9fff]) +(?=[\u3400-\u9fff])', '', value)
            b.text = value.strip()
        result.append(b)
    return result


def fix_headings(blocks: list[Block], *, cfg=None) -> list[Block]:
    cfg = _cfg(cfg)
    current, result = 1, []
    for block in blocks:
        b = block.model_copy(deep=True)
        if b.type == 'heading':
            value = b.text.strip()
            if len(value) > cfg.heading_demote_chars and value.endswith(('。', '.')):
                b.type, b.level = 'paragraph', 0
            else:
                if re.match(r'^第[一二三四五六七八九十百千万零〇\d]+(?:章|部分)', value):
                    level = 1
                elif re.match(r'^第[一二三四五六七八九十百千万零〇\d]+节|^[一二三四五六七八九十]+、', value):
                    level = 2
                elif match := re.match(r'^(\d+(?:\.\d+)*)(?:\.(?=\s|$)|\s|$|(?=[^\d.]))', value):
                    level = match[1].count('.')+1
                elif re.match(r'^[（(][一二三四五六七八九十百\d]+[）)]', value):
                    level = current+1
                else:
                    level = b.level or 1
                b.level = current = min(6, max(1, level))
        result.append(b)
    return result


def remove_repeated(blocks: list[Block], *, cfg=None) -> list[Block]:
    cfg = _cfg(cfg)
    pages = {b.page for b in blocks if b.page is not None}
    if len(pages) < 2:
        return [b.model_copy(deep=True) for b in blocks]
    occurrences = defaultdict(set)
    for b in blocks:
        if b.type in {'paragraph', 'heading'} and b.page is not None and len(b.text) <= cfg.heading_demote_chars:
            occurrences[b.text].add(b.page)
    repeated = {text for text, present in occurrences.items() if len(present) > 1 and
                len(present)/len(pages) > cfg.repeat_page_ratio}
    return [b.model_copy(deep=True) for b in blocks if not (b.type in {'paragraph', 'heading'} and b.text in repeated)]


_TOC = re.compile(r'^.+?(?:\.{3,}|…{2,}|·{3,})\s*\d+\s*$')


_TOC_MARKDOWN = MarkdownIt('commonmark')


def _anchor_line(line: str) -> bool:
    value = re.sub(r'^(?:[-+*]\s+|\d+[.)]\s+)', '', line.strip())
    children = _TOC_MARKDOWN.parseInline(value)[0].children or []
    return bool(children and children[0].type == 'link_open' and children[-1].type == 'link_close'
                and sum(t.type == 'link_open' for t in children) == 1
                and not any(t.type == 'image' for t in children)
                and (children[0].attrGet('href') or '').startswith('#'))


def remove_toc(blocks: list[Block]) -> list[Block]:
    # 本地目录可夹有分组标题/简介：只删纯锚点链接和明确目录标题，保留说明。
    labels = {'目录', '目录导航', 'contents', 'table of contents', 'toc'}
    excluded = set()
    for i, b in enumerate(blocks):
        if b.type != 'heading' or b.text.strip().casefold() not in labels:
            continue
        j, links, headings, anchors = i+1, 0, {i}, set()
        while j < len(blocks):
            current = blocks[j]
            if current.type == 'heading':
                if current.text.strip().casefold() in labels:
                    headings.add(j)
                elif current.level <= b.level:
                    break
                j += 1
                continue
            lines = [line.strip() for line in current.text.splitlines() if line.strip()]
            if current.type not in {'paragraph', 'list'} or not lines:
                break
            pure = all(_anchor_line(line) for line in lines)
            if not pure:
                # 锚点清单后的正文即边界；混合正文永远不删除。
                if links or any(_anchor_line(line) for line in lines):
                    break
            else:
                links += len(lines)
                anchors.add(j)
            j += 1
        if links >= 3:
            excluded.update(headings | anchors)
    blocks = [b for i, b in enumerate(blocks) if i not in excluded]
    grouped = defaultdict(list)
    for b in blocks:
        grouped[b.page].append(b)
    remove = set()
    for page, group in grouped.items():
        lines = [line.strip() for b in group if b.type in {'paragraph', 'heading', 'list'} for line in b.text.splitlines() if line.strip()]
        matches = sum(bool(_TOC.match(line)) for line in lines)
        if matches >= 3 and matches >= len(lines)/2:
            remove.add(page)
    return [b.model_copy(deep=True) for b in blocks if not (b.page in remove and
            (b.page is not None or _TOC.match(b.text.strip()) or b.text.strip() in {'目录', 'Contents'}))]


def _table_rows(block):
    if block.meta.get('format') == 'html':
        return None
    lines = [x.strip() for x in block.text.splitlines() if x.strip()]
    if len(lines) < 2:
        return None
    rows = [re.split(r'(?<!\\)\|', x.strip('|')) for x in lines]
    if not all(re.fullmatch(r'\s*:?-+:?\s*', cell) for cell in rows[1]):
        return None
    return rows, lines


def merge_tables(blocks: list[Block]) -> list[Block]:
    result = []
    for block in blocks:
        b = block.model_copy(deep=True)
        prev = result[-1] if result else None
        if prev and b.type == prev.type == 'table' and prev.page is not None and b.page is not None:
            end = prev.meta.get('page_end', prev.page)
            a, c = _table_rows(prev), _table_rows(b)
            if b.page == end+1 and a and c and len(a[0][0]) == len(c[0][0]):
                same = [x.strip() for x in a[0][0]] == [x.strip() for x in c[0][0]]
                if same or b.meta.get('has_header') is False:
                    prev.text += '\n' + '\n'.join(c[1][2:])
                    prev.meta['page_end'] = b.page
                    for key in ('table_caption', 'table_footnote'):
                        if b.meta.get(key) and b.meta[key] != prev.meta.get(key):
                            prev.meta[key] = '\n'.join(x for x in (prev.meta.get(key, ''), b.meta[key]) if x)
                    continue
        result.append(b)
    return result


def check_equations(blocks: list[Block]) -> list[Block]:
    result = []
    for block in blocks:
        b = block.model_copy(deep=True)
        if b.type == 'equation':
            stack, invalid = [], False
            for match in re.finditer(r'\\.|[{}()\[\]]', b.text):
                char = match[0]
                if char.startswith('\\'):
                    continue
                if char in '{([':
                    stack.append(char)
                elif not stack or stack.pop() != {'}': '{', ')': '(', ']': '['}[char]:
                    invalid = True
            if invalid or stack:
                b.meta['latex_invalid'] = True
            else:
                b.meta.pop('latex_invalid', None)
        result.append(b)
    return result


def fix_ppt(blocks: list[Block], *, filename: str, cfg=None) -> list[Block]:
    result = [Block(type='heading', text=filename, level=1)]
    titled = set()
    for block in blocks:
        b = block.model_copy(deep=True)
        if b.page not in titled and (b.type == 'heading' or (b.type == 'paragraph' and len(b.text) <= _cfg(cfg).heading_demote_chars and not b.text.endswith(('。', '.')))):
            b.type, b.level = 'heading', 2
            titled.add(b.page)
        elif b.type == 'heading':
            b.level = max(3, b.level)
        result.append(b)
    return result


def remove_noise(blocks: list[Block], *, cfg=None) -> list[Block]:
    cfg = _cfg(cfg)
    result = []
    for block in blocks:
        value = block.text.strip()
        if block.type in {'paragraph', 'heading', 'list'}:
            if value in cfg.watermarks or re.fullmatch(r'(?:第\s*)?\d+\s*(?:页|/\s*\d+)?', value):
                continue
            if not any(unicodedata.category(c)[0] in {'L', 'N'} for c in value):
                continue
        result.append(block.model_copy(deep=True))
    return result


def clean_blocks(blocks: list[Block], *, cfg=None, file_type='', filename='') -> list[Block]:
    cfg = _cfg(cfg)
    blocks = remove_toc(blocks)
    blocks = normalize(blocks)
    blocks = fix_headings(blocks, cfg=cfg)
    blocks = remove_repeated(blocks, cfg=cfg)
    blocks = remove_toc(blocks)
    blocks = remove_noise(blocks, cfg=cfg)
    blocks = merge_tables(blocks)
    blocks = check_equations(blocks)
    if file_type in {'ppt', 'pptx'}:
        blocks = fix_ppt(blocks, filename=filename, cfg=cfg)
    return blocks
