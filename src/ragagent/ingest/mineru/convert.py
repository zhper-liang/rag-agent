"""content_list→Block；HTML表格展开span后再转Markdown。"""
from html.parser import HTMLParser
import json
import logging
from pathlib import Path
import re

from ragagent.config import load_config
from ragagent.ingest.models import Block
from ragagent.ingest.images import copy_image, local_image

log = logging.getLogger(__name__)


class _Table(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.row, self.cell = [], None, None
        self.depth = 0
        self.nested = False
        self.header = False
        self.spans = (1, 1)

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.depth += 1
            self.nested |= self.depth > 1
        if self.depth != 1:
            return
        if tag == 'tr':
            self.row = []
        elif tag in {'td', 'th'}:
            self.cell = []
            self.header |= tag == 'th'
            attr = dict(attrs)
            self.spans = tuple(int(attr.get(k, 1)) for k in ('rowspan', 'colspan'))
            if not all(1 <= x <= 10000 for x in self.spans):
                raise ValueError('非法表格span')
        elif tag == 'br' and self.cell is not None:
            self.cell.append('<br>')

    def handle_endtag(self, tag):
        if self.depth == 1:
            if tag in {'td', 'th'} and self.cell is not None and self.row is not None:
                self.row.append((''.join(self.cell).strip(), *self.spans))
                self.cell = None
            elif tag == 'tr' and self.row is not None:
                self.rows.append(self.row)
                self.row = None
        if tag == 'table':
            self.depth -= 1

    def handle_data(self, data):
        if self.depth == 1 and self.cell is not None:
            self.cell.append(data)


def html_table(value: str, *, max_cols: int | None = None) -> tuple[str, dict]:
    max_cols = max_cols if max_cols is not None else load_config().ingest.clean.table_max_cols
    if not re.search(r'<table\b', value, re.I):
        return value, {'format': 'markdown', 'has_header': True}
    table = _Table()
    fallback = (value, {'format': 'html'})
    try:
        table.feed(value)
        if table.nested or not table.rows:
            return fallback
        grid = {}
        for r, row in enumerate(table.rows):
            c = 0
            for text, rowspan, colspan in row:
                while (r, c) in grid:
                    c += 1
                if c+colspan > max_cols or r+rowspan > 10000 or len(grid)+rowspan*colspan > 100000:
                    return fallback
                for dr in range(rowspan):
                    for dc in range(colspan):
                        if (r+dr, c+dc) in grid:
                            return fallback
                        grid[r+dr, c+dc] = text.replace('|', r'\|').replace('\n', '<br>')
                c += colspan
        cols = max(c for _, c in grid)+1
        rows = [[grid.get((r, c), '') for c in range(cols)] for r in range(max(r for r, _ in grid)+1)]
        if not table.header:
            rows.insert(0, ['']*cols)  # 空表头，但meta明确它不是解析出的真实表头。
        rendered = ['| '+' | '.join(row)+' |' for row in rows]
        rendered.insert(1, '| '+' | '.join(['---']*cols)+' |')
        return '\n'.join(rendered), {'format': 'markdown', 'has_header': table.header, 'columns': cols}
    except (ValueError, IndexError):
        log.warning('HTML表格无法安全展开，保留原文')
        return fallback


def _text(value):
    return '\n'.join(map(str, value)) if isinstance(value, list) else str(value or '')


def content_list_to_blocks(items: list[dict], *, max_cols=None) -> list[Block]:
    blocks = []
    for item in items:
        kind = item.get('type', 'text')
        page_idx = item.get('page_idx')
        page = int(page_idx)+1 if page_idx is not None else None
        text, meta, level = _text(item.get('text')), {}, 0
        image = item.get('img_path') or item.get('image_path')
        if kind == 'text':
            level = int(item.get('text_level') or 0)
            kind = 'heading' if level > 0 else 'paragraph'
            level = min(6, level)
        elif kind == 'table':
            text, meta = html_table(_text(item.get('table_body') or item.get('text')), max_cols=max_cols)
            for key in ('table_caption', 'table_footnote'):
                if item.get(key):
                    meta[key] = _text(item[key])
        elif kind == 'image':
            text = _text(item.get('image_caption') or item.get('img_caption') or item.get('caption') or text)
        elif kind == 'list':
            text = _text(item.get('list_items') or text)
        elif kind == 'equation':
            text = _text(item.get('latex') or item.get('text'))
        elif kind == 'code':
            body = _text(item.get('code_body') or item.get('text'))
            lang = _text(item.get('language'))
            fence = '`'*max(3, 1+max((len(m[0]) for m in re.finditer(r'`+', body)), default=0))
            text = body if re.match(r'^(`{3,}|~{3,})', body) else f'{fence}{lang}\n{body}\n{fence}'
            meta['language'] = lang
        else:
            log.warning('未知MinerU类型 %r，保留为paragraph', kind)
            kind = 'paragraph'
        blocks.append(Block(type=kind, text=text, level=level, page=page, image_path=image, meta=meta))
    return blocks


def convert(parsed_dir: str | Path, out_dir: str | Path, *, max_cols=None) -> list[Block]:
    parsed_dir, out_dir = Path(parsed_dir), Path(out_dir)
    lists = sorted(parsed_dir.rglob('*content_list.json'))
    if len(lists) != 1:
        raise ValueError(f'解析产物应只有一个content_list.json，实际{len(lists)}')
    blocks = content_list_to_blocks(json.loads(lists[0].read_text(encoding='utf-8')), max_cols=max_cols)
    for b in blocks:
        if b.image_path:
            try:
                image = local_image(b.image_path, base=lists[0].parent, root=parsed_dir)
                b.image_path = copy_image(image, out_dir)
            except (OSError, ValueError):
                log.warning('MinerU图片引用不可用，去掉路径')
                b.image_path = None
    return blocks
