"""D-014 A：空叶节/空标题链可丢弃，非空祖先和所有正文必须保留。"""
import pytest
from hypothesis import given, settings, strategies as st
from ragagent.ingest.models import Block
from ragagent.ingest.chunking.pipeline import run_chunk, restore_text
from tests.chunking.test_properties import canonical, config, document


def heading(text, level=1):
    return Block(type='heading',text=text,level=level)


@pytest.mark.parametrize('blocks,expected',[
    ([heading('空节'),heading('有内容'),Block(type='paragraph',text='正文')], '# 有内容\n\n正文\n\n'),
    ([heading('父'),heading('子',3),Block(type='paragraph',text='正文')], '# 父\n\n### 子\n\n正文\n\n'),
    ([heading('父'),heading('空子',3),heading('有内容'),Block(type='paragraph',text='正文')], '# 有内容\n\n正文\n\n'),
    ([heading('有内容'),Block(type='paragraph',text='正文'),heading('尾部空节')], '# 有内容\n\n正文\n\n'),
    ([heading('空父'),heading('空子',3),heading('空孙',6)], ''),
    ([heading('父'),Block(type='paragraph',text='父正文'),heading('空子',3)], '# 父\n\n父正文\n\n'),
    ([Block(type='paragraph',text='无标题正文'),heading('空尾',6)], '无标题正文\n\n'),
])
def test_empty_heading_boundaries_keep_every_nonempty_subtree(blocks,expected):
    before=document(blocks).model_dump()
    chunks=run_chunk(document(blocks),cfg=config(),count=len)
    assert canonical(blocks)==expected
    assert restore_text(chunks)==expected
    assert document(blocks).model_dump()==before
    assert all(c.tokens<=config().max_tokens for c in chunks)


@settings(max_examples=120,deadline=None)
@given(st.lists(st.tuples(st.integers(1,6),st.booleans()),min_size=1,max_size=20))
def test_arbitrary_empty_heading_chains_and_siblings(spec):
    blocks=[]
    for i,(level,has_body) in enumerate(spec):
        blocks.append(heading(f'标题{i}',level))
        if has_body:
            blocks.append(Block(type='paragraph',text=f'第{i}段正文'))
    chunks=run_chunk(document(blocks),cfg=config(),count=len)
    assert restore_text(chunks)==canonical(blocks)
    # 每个正文只出现一次；空标题能否恢复不放宽其它正文判定。
    for i,(_,has_body) in enumerate(spec):
        if has_body:
            assert restore_text(chunks).count(f'第{i}段正文\n\n')==1
