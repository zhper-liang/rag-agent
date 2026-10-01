"""Markdown 渲染及 §6 的结构绑定。仅变更流水线内部的 Block 副本。"""
import re

from ragagent.ingest.models import Block
from .models import Unit


def code_parts(block: Block) -> tuple[str, str, str]:
    text = block.text
    m = re.fullmatch(r'(`{3,}|~{3,})([^\n]*)\n([\s\S]*?)\n\1\s*', text)
    if m:
        return m[3] + '\n', m[2], m[1]
    lang = block.meta.get('lang', '')
    # 避免原代码中的围栏字符被 Markdown 误认成结束围栏。
    longest = max([len(m[0]) for m in re.finditer(r'`+', text)] or [0])
    return text + ('' if text.endswith('\n') else '\n'), lang, '`' * max(3, longest + 1)


def render_block(block: Block) -> str:
    if '_rendered' in block.meta:
        return block.meta['_rendered']
    if block.type == 'heading':
        text = '#' * block.level + ' ' + block.text
    elif block.type == 'code':
        body, lang, fence = code_parts(block)
        text = f'{fence}{lang}\n{body}{fence}'
    elif block.type == 'image':
        text = f'[图: {block.text}]'
    elif block.type == 'table':
        caption = block.meta.get('table_caption', '')
        footnote = block.meta.get('table_footnote', '')
        if isinstance(footnote, list):
            footnote = '\n'.join(footnote)
        text = (caption + '\n' if caption else '') + block.text + ('\n' + footnote if footnote else '')
    else:
        text = block.text
    return text + '\n\n'


def prepare_blocks(blocks: list[Block]) -> list[Block]:
    prepared, pos = [], 0
    for i, original in enumerate(blocks):
        b = original.model_copy(deep=True)
        text = render_block(b)
        b.meta.update(_rendered=text, _source_idx=i, _span=(pos, pos + len(text)))
        prepared.append(b)
        pos += len(text)
    return prepared


def unit_text(unit: Unit) -> str:
    return ''.join(render_block(b) for b in unit.blocks)


def build_units(blocks: list[Block], cfg, count) -> list[Unit]:
    units: list[Unit] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        group = [b]
        # 冒号引导列表/表格；公式与两侧解释段落成不可拆 Unit。
        if b.type == 'paragraph' and i + 1 < len(blocks):
            nxt = blocks[i + 1]
            if (b.text.rstrip().endswith((':', '：')) and nxt.type in ('list', 'table')) or nxt.type == 'equation':
                group.append(nxt)
                if nxt.type == 'equation' and i + 2 < len(blocks) and blocks[i + 2].type == 'paragraph':
                    group.append(blocks[i + 2])
        elif b.type == 'equation' and i + 1 < len(blocks) and blocks[i + 1].type == 'paragraph':
            group.append(blocks[i + 1])
        if count(''.join(render_block(x) for x in group)) > cfg.max_tokens:
            group = [b]
        kind = b.type if b.type in ('code', 'table', 'list', 'equation') else 'text'
        text = ''.join(render_block(x) for x in group)
        units.append(Unit(blocks=group, kind=kind, tokens=count(text),
                          hard_break=kind == 'code' and count(text) > cfg.inline_code_tokens))
        i += len(group)
    return units
