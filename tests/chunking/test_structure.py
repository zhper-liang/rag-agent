"""结构策略与边界回归：不是只有六性质才算完整切分实现。"""
import pytest
from ragagent.ingest.models import Block
from ragagent.ingest.chunking.pipeline import run_chunk, restore_text, statistics
from ragagent.ingest.chunking.units import build_units, prepare_blocks, render_block
from ragagent.ingest.chunking.tree import build_tree
from ragagent.ingest.chunking.semantic import boundaries
from ragagent.ingest.chunking.dedup import mark_boilerplate
from tests.chunking.test_properties import config, document, canonical


def run(blocks,cfg=None,encode=None):
    return run_chunk(document(blocks),cfg=cfg or config(),count=len,encode=encode)


def test_paragraph_list_binding_and_equation_explanation():
    blocks = prepare_blocks([Block(type='paragraph',text='参数：'),Block(type='list',text='- one\n- two'),
        Block(type='paragraph',text='定义为'),Block(type='equation',text='$$x=1$$'),
        Block(type='paragraph',text='其中x表示参数')])
    units = build_units(blocks,config(),len)
    assert [len(u.blocks) for u in units] == [2,3]
    assert run([b.model_copy(update={'meta':{}}) for b in blocks])[0].meta['strategy']=='section'


def test_binding_is_released_above_max():
    blocks = prepare_blocks([Block(type='paragraph',text='x'*110+'：'),Block(type='list',text='- abcd'*5)])
    assert [len(u.blocks) for u in build_units(blocks,config(),len)] == [1,1]


def test_tree_skipped_levels_and_section_whole():
    blocks = [Block(type='heading',text='父',level=1),Block(type='heading',text='子',level=4),Block(type='paragraph',text='正文')]
    tree = build_tree(prepare_blocks(blocks),len)
    assert tree.children[0].children[0].path == ['父','子']
    c = run(blocks)
    assert len(c)==1 and c[0].meta['strategy']=='section'
    assert restore_text(c)==canonical(blocks)


def test_leaf_kept_between_target_and_max():
    chunks = run([Block(type='heading',text='节',level=2),Block(type='paragraph',text='a'*80)])
    assert len(chunks)==1 and chunks[0].meta['strategy']=='section'
    assert chunks[0].section_path == ['节']
    assert chunks[0].embed_text.count('## 节')==0


def test_big_sibling_sections_do_not_mix():
    blocks = [Block(type='heading',text='A',level=1),Block(type='paragraph',text='a'*150),
              Block(type='heading',text='B',level=1),Block(type='paragraph',text='b'*150)]
    chunks = run(blocks)
    assert restore_text(chunks)==canonical(blocks)
    assert all(not ('aaaa' in c.text and 'bbbb' in c.text) for c in chunks)
    assert {tuple(c.section_path) for c in chunks}=={('A',),('B',)}


def test_code_lead_in_links_pages_and_raw_body():
    blocks = [Block(type='paragraph',text='先准备环境。执行以下命令。',page=3),
              Block(type='code',text='echo hello\n'*8,meta={'lang':'bash'},page=4)]
    chunks = run(blocks)
    code = [c for c in chunks if c.type=='code']
    assert code and all(c.meta['lead_in']=='执行以下命令。' for c in code)
    assert code[0].related_ids == [chunks[0].chunk_id]
    assert code[0].chunk_id in chunks[0].related_ids
    assert chunks[0].page_start==3 and code[0].page_start==4
    assert code[0].prev_id==chunks[0].chunk_id
    assert restore_text(chunks)==canonical(blocks)


def test_embedded_code_fences_use_longer_fence():
    blocks = [Block(type='code',text='print("```")\n'*15,meta={'lang':'python'})]
    chunks = run(blocks)
    assert all(c.text.startswith('````python\n') and c.text.endswith('````\n\n') for c in chunks)
    assert restore_text(chunks)==render_block(blocks[0])


def test_table_headers_repeated_and_caption_preserved():
    table = '| A | B |\n|---|---|\n'+''.join(f'| row{i} | {i} |\n' for i in range(20))
    b = Block(type='table',text=table,meta={'table_caption':'参数表','table_footnote':'脚注'})
    chunks = run([b])
    assert len(chunks)>1
    assert all('| A | B |\n|---|---|' in c.text for c in chunks)
    assert all('参数表' in c.embed_text for c in chunks)
    assert restore_text(chunks)==render_block(b)


def test_list_split_top_items():
    block = Block(type='list',text=''.join(f'- item{i}\n  continued\n' for i in range(20)))
    chunks = run([block])
    assert all(c.text.startswith('- item') for c in chunks)
    assert restore_text(chunks)==canonical([block])


def test_semantic_zscore_strong_break_and_uniform():
    cfg = config().semantic.model_copy(update={'enabled':True,'break_z':-1,'strong_z':1})
    values = [[1,0],[1,0],[0,1],[0,1],[1,0]]
    result = boundaries(['a','b','c','d','e'],cfg,lambda _:values)
    assert result[1]=='strong' and result[2]=='break'
    assert boundaries(['a']*5,cfg,lambda _: [[1,0]]*5)=={}
    assert boundaries(['a']*3,cfg,lambda _: pytest.fail('short section should not embed'))=={}


def test_semantic_only_embeds_oversized_sections():
    sem = config().semantic.model_copy(update={'enabled':True})
    cfg = config().model_copy(update={'semantic':sem})
    run([Block(type='paragraph',text='small')],cfg=cfg,encode=lambda _:pytest.fail('small whole section embedded'))
    calls=[]
    blocks = [Block(type='paragraph',text=x*26) for x in 'abcdefgh']
    chunks = run(blocks,cfg=cfg,encode=lambda texts: calls.append(texts) or [[1,0] if i<4 else [0,1] for i in range(len(texts))])
    assert len(calls)==1 and any(c.meta['strategy']=='semantic' for c in chunks)
    assert restore_text(chunks)==canonical(blocks)


def test_dedup_marks_not_deletes_and_boilerplate_is_cross_doc():
    blocks = [Block(type='paragraph',text='same'*12),Block(type='paragraph',text='same'*12),Block(type='paragraph',text='different'*10)]
    chunks = run(blocks)
    assert len(chunks)==3 and chunks[0].searchable and not chunks[1].searchable
    assert chunks[1].prev_id==chunks[0].chunk_id and chunks[1].next_id==chunks[2].chunk_id
    other = run(blocks)
    for c in other:
        c.doc_id='fedcba9876543210'
    mark_boilerplate([chunks,other],threshold=1)
    assert not any(c.searchable for c in chunks+other)


def test_images_not_text_chain_and_screenshots_not_image_chunks():
    chunks = run([Block(type='paragraph',text='before'),Block(type='image',text='caption',image_path='i.jpg'),
        Block(type='table',text='| a |\n|---|\n| b |',image_path='table.jpg'),Block(type='paragraph',text='after')])
    images = [c for c in chunks if c.type=='image']
    assert len(images)==1 and images[0].prev_id is None and images[0].next_id is None
    assert any('table.jpg' in c.meta.get('image_paths',[]) for c in chunks)
    assert statistics(chunks,config())['max']<=config().max_tokens


def test_empty_document_and_lone_heading():
    assert run([])==[]
    assert run([Block(type='heading',text='empty',level=1)])==[]
