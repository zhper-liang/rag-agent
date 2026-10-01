"""清洗八规则手写快照；同时证明输入不被原地修改。"""
import json
from pathlib import Path
import pytest
from ragagent.config import load_config
from ragagent.ingest.models import Block
from ragagent.ingest import cleaners

SNAPSHOTS = json.loads((Path(__file__).parent/'fixtures'/'clean_snapshots.json').read_text())


def snap(name, actual):
    assert [b.model_dump() for b in actual] == SNAPSHOTS[name]


def b(kind='paragraph', text='', **kw):
    return Block(type=kind, text=text, **kw)


def test_encoding_snapshot():
    blocks = [b(text='ＡＢＣ\u200b\x01  中 文\n换行'), b('code', '```\n  Ａ\u200b\n```'), b('equation', '\\frac{Ａ}{ B }'),
              b('list', '- 一级\n  - 二级'), b('table', '| 表 | 空 |\n| --- | --- |\n| 值 |  |')]
    before = [x.model_dump() for x in blocks]
    snap('encoding', cleaners.normalize(blocks))
    assert [x.model_dump() for x in blocks] == before


def test_heading_snapshot():
    cfg = load_config().ingest.clean.model_copy(update={'heading_demote_chars': 10})
    blocks = [b('heading', s, level=1) for s in ['第一章 起始', '第一节 概述', '一、方法', '1. 总览', '1.1 设计', '1.1.1 细节', '（一）子级', '无编号', '这其实是一段超过十字的完整句子。']]
    snap('headings', cleaners.fix_headings(blocks, cfg=cfg))


def test_repeated_headers_snapshot():
    blocks = [b(text='公司保密', page=i) for i in range(1, 4)]+[b(text=f'正文不同{i}', page=i) for i in range(1, 4)]
    snap('repeat', cleaners.remove_repeated(blocks))
    assert cleaners.remove_repeated([b(text='唯一正文', page=1)]) == [b(text='唯一正文', page=1)]
    # 相同页面重复不按页面重复删除。
    assert len(cleaners.remove_repeated([b(text='文本', page=1), b(text='文本', page=1), b(text='其他', page=2)])) == 3


def test_toc_snapshot():
    blocks = [b('heading', '目录', level=1, page=1), b(text='概述……1\n方法……2\n结论……3', page=1), b(text='正文完整内容', page=2)]
    snap('toc', cleaners.remove_toc(blocks))
    snap('toc', cleaners.clean_blocks(blocks))
    assert cleaners.remove_toc([b(text='正文提到参考……1', page=2)]) == [b(text='正文提到参考……1', page=2)]


def test_cross_page_tables_snapshot():
    blocks = [b('table', '| A | B |\n| --- | --- |\n| 1 | 2 |', page=1, meta={'has_header': True, 'table_caption': '表一'}),
              b('table', '| A | B |\n| --- | --- |\n| 3 | 4 |', page=2, meta={'has_header': True}),
              b('table', '|  |  |\n| --- | --- |\n| 5 | 6 |', page=3, meta={'has_header': False, 'table_footnote': '脚注'})]
    snap('tables', cleaners.merge_tables(blocks))
    different = b('table', '| C | D |\n| --- | --- |\n| 7 | 8 |', page=2)
    assert len(cleaners.merge_tables([blocks[0], different])) == 2
    assert len(cleaners.merge_tables([blocks[0], blocks[1].model_copy(update={'page': 3})])) == 2


def test_formula_snapshot():
    snap('equations', cleaners.check_equations([b('equation', r'\frac{a}{b}+(c)'), b('equation', r'\frac{a}{b'), b('equation', r'\{a\}')]))


def test_ppt_snapshot():
    snap('ppt', cleaners.fix_ppt([b('heading', '第一页', level=1, page=1), b(text='页面正文', page=1), b('heading', '第二页', level=1, page=2)], filename='演示'))


def test_noise_snapshot():
    cfg = load_config().ingest.clean.model_copy(update={'watermarks': ['水印']})
    snap('noise', cleaners.remove_noise([b(text='水印'), b(text='---'), b(text='12'), b(text='第 2 页'), b(text='有效正文'), b('equation', '---')], cfg=cfg))


def test_ppt_missing_title_level_can_promote_first_short_text():
    result=cleaners.fix_ppt([b(text='幻灯片标题',page=1),b(text='下面是正常的一段正文。',page=1)],filename='演示')
    assert result[1].type=='heading' and result[1].level==2 and result[2].type=='paragraph'
    assert cleaners.fix_headings([b('heading','1.1.1',level=1)])[0].level==3


def test_explicit_markdown_anchor_toc_snapshot():
    blocks = [b('heading', '目录导航', level=1), b('heading', '目录导航', level=1),
              b('list', '- [安装](#install)\n  - [校验](#verify)\n- [排查](#debug)'),
              b('heading', '合成维护手册', level=1), b(text='这里是需要保留的正文。')]
    snap('markdown_anchor_toc', cleaners.remove_toc(blocks))
    # 无显式目录标题的链接清单和混合正文不可按目录删掉。
    assert cleaners.remove_toc(blocks[2:]) == blocks[2:]
    mixed = [b('heading', '目录', level=1), b(text='[一步](#a)\n[二步](#b)\n[三步](#c)\n这里是真正的正文。')]
    assert cleaners.remove_toc(mixed) == mixed
    assert cleaners.remove_toc(blocks[:3]) == []


def test_toc_interleaved_headings_and_intro_preserved():
    blocks = [b('heading', '目录', level=1), b('heading', '合成指南', level=2),
              b(text='这段是简介，不能随目录删除。'),
              b('heading', '基础操作', level=3),
              b(text='[安装](<#install>)'), b('list', '- [校验](#verify)\n- [维护](<#maintain>)'),
              b('heading', '安装', level=2), b(text='这里开始真正的操作正文。'),
              b('list', '- [附录一](#appendix1)\n- [附录二](#appendix2)\n- [附录三](#appendix3)')]
    cleaned = cleaners.remove_toc(blocks)
    assert cleaned == [blocks[i] for i in (1,2,3,6,7,8)]


def test_mixed_anchor_links_are_not_a_toc():
    blocks = [b('heading', '目录', level=1), b('list',
        '- [部署](#deploy) 参见 [官网](https://example.com)\n'
        '- [维护](#ops) 这是需要保留的正文 [更多](#more)\n'
        '- [校验](#verify) 引用 [外部](https://example.com/verify)')]
    assert cleaners.remove_toc(blocks) == blocks
