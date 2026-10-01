"""markdown-it-py AST→Block，结构块保留原文，不用正则重建 Markdown。"""
from html.parser import HTMLParser
import logging
from pathlib import Path
import re

from markdown_it import MarkdownIt
from ragagent.ingest.images import copy_image, local_image
from ragagent.ingest.models import Block

log = logging.getLogger(__name__)


class _HeadingText(HTMLParser):
    """仅提取独立HTML标题的文字，不执行HTML，也不改写混合HTML正文。"""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def _html_heading(value):
    match = re.fullmatch(r'\s*<h([1-6])\b[^>]*>(.*?)</h\1>\s*', value, re.I | re.S)
    if not match or re.search(r'</?h[1-6]\b', match[2], re.I):
        return None
    parser = _HeadingText()
    parser.feed(match[2])
    return Block(type='heading', text=''.join(parser.parts).strip(), level=int(match[1]))


def _strip_image_markup(value, images):
    """匹配AST已确认的图片，兼容引用式语法、转义及URL中的括号。"""
    def close(start, opening, closing):
        depth, escaped = 1, False
        for j in range(start+1, len(value)):
            char = value[j]
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if not depth:
                    return j
        return None
    ranges, cursor = [], 0
    for image in images:
        while (start := value.find('![', cursor)) >= 0:
            end = close(start+1, '[', ']')
            if end is None:
                break
            label = value[start+2:end]
            cursor = end+1
            if re.sub(r'\\(.)', r'\1', label) != image.content:
                continue
            if cursor < len(value) and value[cursor] in '([':
                last = close(cursor, value[cursor], ')' if value[cursor] == '(' else ']')
                if last is not None:
                    cursor = last+1
            ranges.append((start, cursor))
            break
    for start, end in reversed(ranges):
        value = value[:start]+value[end:]
    return value


def _images(inline, path, root, out_dir):
    result = []
    for image in (t for t in inline.children or [] if t.type == 'image'):
        try:
            source = local_image(image.attrGet('src') or '', base=path.parent, root=root)
            copied = copy_image(source, out_dir)
            result.append(Block(type='image', text=image.content, image_path=copied))
        except (OSError, ValueError):
            log.warning('忽略远端/越界/缺失图片 path=%s', path.name)
    return result


def parse_markdown(path: str | Path, out_dir: str | Path, *, input_root=None) -> list[Block]:
    path, out_dir = Path(path), Path(out_dir)
    root = Path(input_root) if input_root is not None else path.parent
    text = path.read_text(encoding='utf-8-sig')
    lines = text.splitlines(keepends=True)
    tokens = MarkdownIt('commonmark').enable('table').parse(text)
    blocks, i = [], 0
    while i < len(tokens):
        token = tokens[i]
        # map含物理行；外层list/table一次取完，内部token不重复输出。
        if token.type in {'bullet_list_open', 'ordered_list_open', 'table_open'}:
            kind = 'table' if token.type == 'table_open' else 'list'
            start, end = token.map
            block = Block(type=kind, text=''.join(lines[start:end]).rstrip('\r\n'))
            if kind == 'table':
                block.meta['has_header'] = True
            j = i + 1
            while j < len(tokens) and not (tokens[j].level == token.level and tokens[j].type == token.type.replace('_open', '_close')):
                j += 1
            inlines = [t for t in tokens[i:j] if t.type == 'inline']
            images = [child for t in inlines for child in t.children or [] if child.type == 'image']
            block.text = _strip_image_markup(block.text, images)
            blocks.append(block)
            for inline in inlines:
                blocks.extend(_images(inline, path, root, out_dir))
            i = j + 1
            continue
        if token.type in {'fence', 'code_block'}:
            # 统一成围栏；围栏长度必须比代码内的所有反引号run长。
            fence = '`' * max(3, 1 + max((len(m[0]) for m in re.finditer(r'`+', token.content)), default=0))
            blocks.append(Block(type='code', text=f'{fence}{token.info.strip()}\n{token.content.rstrip(chr(10))}\n{fence}',
                                meta={'language': token.info.strip()}))
        elif token.type == 'heading_open':
            inline = tokens[i+1]
            images = [t for t in inline.children or [] if t.type == 'image']
            blocks.append(Block(type='heading', text=_strip_image_markup(inline.content, images), level=int(token.tag[1:])))
            blocks.extend(_images(inline, path, root, out_dir))
        elif token.type == 'paragraph_open':
            inline = tokens[i+1]
            value = inline.content
            if re.fullmatch(r'\s*\$\$[\s\S]*\$\$\s*', value):
                blocks.append(Block(type='equation', text=value.strip()[2:-2].strip()))
            elif re.fullmatch(r'\s*\\\[[\s\S]*\\\]\s*', value):
                blocks.append(Block(type='equation', text=value.strip()[2:-2].strip()))
            else:
                # 用AST的图片token确定引用，文字/链接保留原Markdown语义。
                images = [t for t in inline.children or [] if t.type == 'image']
                body = _strip_image_markup(value, images)
                if body.strip():
                    blocks.append(Block(type='paragraph', text=body.strip()))
                blocks.extend(_images(inline, path, root, out_dir))
        elif token.type == 'html_block':
            heading = _html_heading(token.content)
            if heading is not None:
                blocks.append(heading)
            elif token.content.lstrip().lower().startswith('<table'):
                from ragagent.ingest.mineru.convert import html_table
                body, meta = html_table(token.content)
                blocks.append(Block(type='table', text=body, meta=meta))
            else:
                blocks.append(Block(type='paragraph', text=token.content.strip()))
        i += 1
    return blocks
