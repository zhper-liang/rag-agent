from pathlib import Path
import pytest
from PIL import Image
from ragagent.ingest.parsers.markdown import parse_markdown
from ragagent.ingest.parsers.text import parse_text
from ragagent.ingest.router import route


@pytest.mark.parametrize('text', ['# 一级\n\n普通 **强调** 与 [链接](https://example.com) 正文。\n',
                                 '一级\n====\n\n> 引用内容\n\n## 二级\n',
                                 '\ufeff# 标题\r\n\r\n末行没有换行'])
def test_markdown_prose_examples(tmp_path,text):
    path = tmp_path/'a.md'
    path.write_text(text)
    blocks = parse_markdown(path,tmp_path/'out')
    assert blocks[0].type == 'heading' and blocks[0].level == 1
    assert blocks[-1].text


def test_markdown_structures_and_images(tmp_path):
    image = tmp_path/'flow.png'
    Image.new('RGB',(128,128),'red').save(image)
    path = tmp_path/'a.md'
    path.write_text('# 标题\n\n- 一级\n  - 二级\n\n|列|值|\n|---|---|\n|a|b|\n\n```python\n  print("Ａ")\n```\n\n$$\n\\frac{a}{b}\n$$\n\n前文![流程](flow.png)后文\n\n![远端](https://x/y.png)\n\n![越界](../secret.png)\n')
    out = tmp_path/'out'
    blocks = parse_markdown(path,out)
    assert [b.type for b in blocks] == ['heading','list','table','code','equation','paragraph','image']
    assert blocks[1].text == '- 一级\n  - 二级'
    assert blocks[3].text == '```python\n  print("Ａ")\n```'
    assert blocks[4].text == r'\frac{a}{b}'
    assert blocks[5].text == '前文后文'
    assert (out/blocks[6].image_path).read_bytes() == image.read_bytes()


def test_symlink_image_cannot_escape_input(tmp_path):
    root = tmp_path/'input'
    root.mkdir()
    outside = tmp_path/'outside.png'
    Image.new('RGB',(100,100)).save(outside)
    (root/'link.png').symlink_to(outside)
    (root/'a.md').write_text('![secret](link.png)')
    assert parse_markdown(root/'a.md',tmp_path/'out') == []


@pytest.mark.parametrize('text,expected',[('第一段文本\n继续本段\n\n第二段文本',['第一段文本\n继续本段','第二段文本']),
                                         ('\ufeff第一段\r\n\r\n  第二段  ',['第一段','第二段']), ('\n\n',[])])
def test_txt_samples(tmp_path,text,expected):
    path=tmp_path/'a.txt'
    path.write_text(text)
    assert [b.text for b in parse_text(path)] == expected


def test_routes_configuration():
    assert route('a.MD') == route('a.txt') == 'text'
    for ext in ('pdf','doc','docx','ppt','pptx','png','jpg','jpeg'):
        assert route('a.'+ext) == 'mineru'
    assert route('a.xlsx') is None and route('a.zip') is None


def test_reference_style_parentheses_and_list_images(tmp_path):
    Image.new('RGB',(100,100),'blue').save(tmp_path/'f(1).png')
    Image.new('RGB',(100,100),'red').save(tmp_path/'ref.png')
    path=tmp_path/'a.md'
    path.write_text('前![说明](f(1).png)后\n\n![引用][key]\n\n- 步骤 ![列表](ref.png)\n\n[key]: ref.png\n')
    blocks=parse_markdown(path,tmp_path/'out')
    assert [b.type for b in blocks]==['paragraph','image','image','list','image']
    assert blocks[0].text=='前后' and blocks[3].text=='- 步骤 '
    assert not any('![' in b.text for b in blocks)


def test_standalone_html_headings_are_structural(tmp_path):
    path = tmp_path/'a.md'
    path.write_text('<h1 id="manual">合成 <em>维护</em> &amp; 排查</h1>\n\n'
                    '<H2 data-id="step">部署步骤</H2>\n\n正文内容\n\n'
                    '<h3>不能吞掉正文</h3><p>同块文字必须保留</p>\n')
    blocks = parse_markdown(path, tmp_path/'out')
    assert [(b.type, b.level, b.text) for b in blocks[:3]] == [
        ('heading', 1, '合成 维护 & 排查'), ('heading', 2, '部署步骤'), ('paragraph', 0, '正文内容')]
    assert blocks[-1].type == 'paragraph' and '同块文字必须保留' in blocks[-1].text
