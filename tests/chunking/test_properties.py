"""文档切分 §13：六条独立性质，混合结构及最坏情况随机输入。"""
import re

from hypothesis import given, settings, strategies as st

from functools import lru_cache
import yaml
from ragagent.config import DEFAULT_PATH, Chunking
from ragagent.ingest.models import Block, CleanDocument
from ragagent.ingest.chunking.pipeline import run_chunk, restore_text

ALPHABET = 'abcdef部署环境参数。！？；，、 \n🙂'
TEXT = st.text(ALPHABET, min_size=1, max_size=220)


def document(blocks):
    return CleanDocument(doc_id='0123456789abcdef', version=1,
        source_path='synthetic.md', file_type='md', parser='text', title='样本文档',
        content_hash='0'*64, blocks=blocks)


@lru_cache
def config():
    cfg = Chunking.model_validate(yaml.safe_load(DEFAULT_PATH.read_text())["chunking"])
    return cfg.model_copy(update={
        'target_tokens': 64, 'max_tokens': 120, 'min_tokens': 12,
        'overlap_tokens': 12, 'inline_code_tokens': 24,
        'semantic': cfg.semantic.model_copy(update={'enabled': False}),
    })


@st.composite
def mixed_blocks(draw):
    out = []
    for i in range(draw(st.integers(1, 14))):
        if draw(st.booleans()):
            out.append(Block(type='heading', text=f'章节{i}', level=draw(st.integers(1,6))))
        kind = draw(st.sampled_from(['paragraph', 'code', 'table', 'list', 'equation', 'image']))
        if kind == 'code':
            text = ''.join(f'x{i} = {s!r}\n' for s in draw(st.lists(TEXT, min_size=1,max_size=4)))
            out.append(Block(type=kind,text=text,meta={'lang':'python'},page=i+1))
        elif kind == 'table':
            rows = draw(st.lists(st.text('abc参数',min_size=1,max_size=80),min_size=1,max_size=5))
            text = '| A | B |\n|---|---|\n' + ''.join(f'| {r} | {i} |\n' for r in rows)
            out.append(Block(type=kind,text=text,page=i+1))
        elif kind == 'list':
            out.append(Block(type=kind,text=''.join(f'- {s}\n' for s in draw(st.lists(TEXT,min_size=1,max_size=4)))))
        elif kind == 'image':
            out.append(Block(type=kind,text=f'图片{i}',image_path=f'images/{i}.jpg',page=i+1))
        else:
            out.append(Block(type=kind,text=draw(TEXT),page=i+1))
    return out


def canonical(blocks):
    """独立 oracle：正文块按 Markdown 渲染，不包含图片占位符。"""
    parts = []
    for b in blocks:
        if b.type == 'image':
            continue
        if b.type == 'heading':
            s = '#' * b.level + ' ' + b.text
        elif b.type == 'code':
            s = '```' + b.meta.get('lang','') + '\n' + b.text + ('' if b.text.endswith('\n') else '\n') + '```'
        else:
            s = b.text
        parts.append(s + '\n\n')
    return ''.join(parts)


@settings(max_examples=120, deadline=None)
@given(mixed_blocks())
def test_tokens_never_exceed_max(blocks):
    chunks = run_chunk(document(blocks), cfg=config(), count=len)
    assert all(c.tokens == len(c.text) <= config().max_tokens for c in chunks)


@settings(max_examples=120, deadline=None)
@given(st.lists(TEXT, min_size=1,max_size=15))
def test_full_coverage_removing_overlap(texts):
    blocks = [Block(type='paragraph',text=s) for s in texts]
    chunks = run_chunk(document(blocks), cfg=config(), count=len)
    assert restore_text(chunks) == canonical(blocks)


@settings(max_examples=100, deadline=None)
@given(st.lists(st.text('abc。',min_size=1,max_size=90),min_size=1,max_size=15))
def test_unit_boundaries_not_split(texts):
    chunks = run_chunk(document([Block(type='paragraph',text=s) for s in texts]), cfg=config(), count=len)
    boundaries, pos = {0}, 0
    for s in texts:
        pos += len(s) + 2
        boundaries.add(pos)
    for c in chunks:
        for start, end in c.meta['source_spans']:
            assert start in boundaries and end in boundaries
        assert c.meta['strategy'] != 'split'


@settings(max_examples=100, deadline=None)
@given(st.lists(st.text('ab (),;{}',min_size=1,max_size=180),min_size=1,max_size=12))
def test_code_fences_and_line_boundaries(lines):
    payload = '\n'.join(lines) + '\n'
    blocks = [Block(type='code',text=payload,meta={'lang':'python'})]
    chunks = run_chunk(document(blocks), cfg=config(), count=len)
    assert restore_text(chunks) == canonical(blocks)
    for c in chunks:
        if c.type != 'code':
            continue
        assert re.fullmatch(r'(`{3,}|~{3,})python\n[\s\S]*\n\1\n\n', c.text)
        if not c.meta.get('force_split'):
            start, end = c.meta['code_body_span']
            assert start == 0 or payload[start-1] == '\n'
            assert end == len(payload) or payload[end-1] == '\n'


@settings(max_examples=100, deadline=None)
@given(st.lists(st.booleans(),min_size=1,max_size=20))
def test_image_anchors_bidirectional(flags):
    blocks = [Block(type='image',text=f'图{i}',image_path=f'images/{i}.jpg') if flag
              else Block(type='paragraph',text=f'段落{i}') for i,flag in enumerate(flags)]
    chunks = run_chunk(document(blocks), cfg=config(), count=len)
    by_id = {c.chunk_id:c for c in chunks}
    images = [c for c in chunks if c.type == 'image']
    assert len(images) == sum(flags)
    for c in images:
        anchor = by_id[c.meta['anchor_id']]
        assert anchor.type == 'text'
        assert c.chunk_id in anchor.image_ids
    for c in chunks:
        for image_id in c.image_ids:
            assert by_id[image_id].meta['anchor_id'] == c.chunk_id


@settings(max_examples=100, deadline=None)
@given(mixed_blocks())
def test_deterministic_output(blocks):
    doc = document(blocks)
    before = doc.model_dump()
    a = run_chunk(doc,cfg=config(),count=len)
    b = run_chunk(doc,cfg=config(),count=len)
    assert [c.model_dump() for c in a] == [c.model_dump() for c in b]
    assert doc.model_dump() == before
    ids = [c.chunk_id for c in a]
    assert len(ids) == len(set(ids))


@settings(max_examples=80, deadline=None)
@given(mixed_blocks())
def test_mixed_source_coverage(blocks):
    chunks = run_chunk(document(blocks),cfg=config(),count=len)
    assert restore_text(chunks) == canonical(blocks)
