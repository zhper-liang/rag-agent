"""真实 bge-m3 tokenizer 文件的验证（没有模型文件必须明确跳过）。"""
from pathlib import Path
import re

import pytest
from hypothesis import given, settings, strategies as st

from ragagent.ingest.chunking.tokenizer import TokenCounter, count_tokens
from ragagent.ingest.chunking.pipeline import run_chunk, restore_text
from ragagent.ingest.models import Block
from tests.chunking.test_properties import document,config,canonical,mixed_blocks

MODEL = Path('models/BAAI/bge-m3/tokenizer.json')
pytestmark = pytest.mark.skipif(not MODEL.is_file(),reason='未下载真实 BAAI/bge-m3 tokenizer.json')


@pytest.fixture(scope='module')
def counter():
    return TokenCounter('BAAI/bge-m3',path=MODEL)


def test_bge_counts_cache_and_no_truncation(counter):
    counter.clear_cache()
    assert counter('')==0
    assert counter('hello world')==3
    sample='部署 Milvus DISKANN 索引🙂'
    assert counter(sample)==count_tokens(sample)
    before=counter.cache_info().hits
    counter(sample)
    assert counter.cache_info().hits==before+1
    assert counter('hello '*10000)>=10000
    with pytest.raises(TypeError):
        counter(None)


@settings(max_examples=100,deadline=None)
@given(mixed_blocks())
def test_real_bge_six_properties(counter,blocks):
    chunks=run_chunk(document(blocks),cfg=config(),count=counter)
    assert all(c.tokens==counter(c.text)<=config().max_tokens for c in chunks)
    assert restore_text(chunks)==canonical(blocks)
    boundaries={0};pos=0
    for b in blocks:
        if b.type=='image':
            pos+=len(f'[图: {b.text}]\n\n')
        elif b.type=='heading':
            # source_spans 使用原始渲染坐标，不能用排除空标题的覆盖 oracle。
            pos+=len('#'*b.level+' '+b.text+'\n\n')
        else:
            pos+=len(canonical([b]))
        boundaries.add(pos)
    for c in chunks:
        if c.type=='image':
            continue
        if c.meta['strategy']!='split':
            assert all(a in boundaries and b in boundaries for a,b in c.meta['source_spans'])
        if c.type=='code':
            assert re.fullmatch(r'(`{3,}|~{3,})python\n[\s\S]*\n\1\n\n',c.text)
    by_id={c.chunk_id:c for c in chunks}
    assert len([c for c in chunks if c.type=='image'])==sum(b.type=='image' for b in blocks)
    for c in chunks:
        if c.type=='image':
            anchor=by_id[c.meta['anchor_id']]
            assert anchor.type=='text' and c.chunk_id in anchor.image_ids
        for image_id in c.image_ids:
            assert by_id[image_id].meta['anchor_id']==c.chunk_id
    second=run_chunk(document(blocks),cfg=config(),count=counter)
    assert [c.model_dump() for c in chunks]==[c.model_dump() for c in second]


@settings(max_examples=80,deadline=None)
@given(st.text('部署🙂abc，。\n ',min_size=1,max_size=400),st.integers(2,12))
def test_real_bge_oversized_unicode_no_loss(counter,text,n):
    blocks=[Block(type='paragraph',text=text*n)]
    chunks=run_chunk(document(blocks),cfg=config(),count=counter)
    assert all(c.tokens==counter(c.text)<=config().max_tokens for c in chunks)
    assert restore_text(chunks)==canonical(blocks)
