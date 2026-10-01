"""§5 标题树：跳级挂最近祖先，先序编号。"""
from .models import Section
from .units import render_block


def build_tree(blocks, count) -> Section:
    root = Section(heading=None, level=0, path=[], idx=0, body=[], children=[])
    stack = [root]
    idx = 0
    for b in blocks:
        if b.type != 'heading':
            stack[-1].body.append(b)
            continue
        while len(stack) > 1 and stack[-1].level >= b.level:
            stack.pop()
        idx += 1
        b.meta['_section_idx'] = idx
        sec = Section(heading=b.text, level=b.level, path=stack[-1].path + [b.text],
                      idx=idx, body=[], children=[])
        stack[-1].children.append(sec)
        stack.append(sec)
    headings = {b.meta['_section_idx']: b for b in blocks if b.type == 'heading'}

    def calculate(sec):
        sec.children = [c for c in sec.children if c.body or c.children]
        if not sec.body and not sec.children:
            sec.tokens = 0
            return ''
        text = (render_block(headings[sec.idx]) if sec.heading is not None else '')
        text += ''.join(render_block(b) for b in sec.body)
        for child in sec.children:
            text += calculate(child)
        sec.tokens = count(text)
        return text
    calculate(root)
    return root
