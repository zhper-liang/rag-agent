import json
from pathlib import Path
import pytest
from ragagent.ingest.mineru.convert import content_list_to_blocks, convert, html_table


def test_handwritten_mapping_snapshot(caplog):
    items = json.loads((Path(__file__).parent/'fixtures'/'handwritten_content_list.json').read_text())
    blocks = content_list_to_blocks(items)
    expected = json.loads((Path(__file__).parent/'fixtures'/'handwritten_blocks.json').read_text())
    assert [b.model_dump() for b in blocks] == expected
    assert [b.type for b in blocks] == ['heading','paragraph','list','table','image','image','equation','code','paragraph']
    assert blocks[0].level == 1 and blocks[1].level == 0
    assert [b.page for b in blocks] == [1,1,1,2,2,2,3,3,3]
    assert blocks[3].text == '| 型号 | 参数 | 参数 |\n| --- | --- | --- |\n| 型号 | 速度 | 容量 |\n| M3 | 10 | 20 |'
    assert blocks[3].meta == {'format':'markdown','has_header':True,'columns':3,'table_caption':'表一：参数','table_footnote':'注：测试'}
    assert blocks[4].text == '流程图说明' and blocks[5].text == '第二图说明'
    assert blocks[6].text == r'\frac{a}{b}' and blocks[6].image_path == 'images/equation.png'
    assert blocks[7].text == '```python\nprint("hello")\n```'
    assert '未知MinerU类型' in caplog.text


@pytest.mark.parametrize('html', [
    '<table><tr><td><table><tr><td>嵌套</td></tr></table></td></tr></table>',
    '<table><tr><td colspan="21">宽表</td></tr></table>',
    '<table><tr><td rowspan="10000" colspan="20">爆炸</td></tr></table>',
    '<table><tr><td colspan="-1">非法</td></tr></table>',
])
def test_complex_tables_preserve_html(html):
    assert html_table(html, max_cols=20) == (html, {'format':'html'})


def test_no_header_span_and_escaped_pipe():
    value, meta = html_table('<table><tr><td>a|b</td><td>x<br>y</td></tr></table>')
    assert value == '|  |  |\n| --- | --- |\n| a\\|b | x<br>y |'
    assert not meta['has_header']


def test_convert_safe_images(tmp_path):
    from PIL import Image
    parsed, out = tmp_path/'parsed', tmp_path/'out'
    (parsed/'images').mkdir(parents=True)
    Image.new('RGB', (128,128),'blue').save(parsed/'images'/'valid.png')
    outside = tmp_path/'outside.png'
    Image.new('RGB',(128,128)).save(outside)
    items = [{'type':'image','img_path':'images/valid.png'}, {'type':'image','img_path':'../outside.png'},
             {'type':'table','table_body':'<table><tr><td>值</td></tr></table>','img_path':'/etc/passwd'}]
    (parsed/'x_content_list.json').write_text(json.dumps(items))
    blocks = convert(parsed,out)
    assert blocks[0].image_path.startswith('images/')
    assert (out/blocks[0].image_path).read_bytes() == (parsed/'images/valid.png').read_bytes()
    assert blocks[1].image_path is None and blocks[2].image_path is None
    with pytest.raises(ValueError):
        convert(tmp_path/'missing',out)
