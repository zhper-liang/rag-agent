import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ragagent.api.app import STATIC
from ragagent.postprocess.offsets import utf16_offset
from tests.integration.helpers import KEYS, api_client, config, deps

NODE = shutil.which('node')
# data: URL 导入，不依赖 package.json 的 type=module 和 node 版本的模块探测。
HARNESS = """
import { readFileSync } from 'node:fs';
const src = readFileSync(process.argv[1], 'utf8');
const mod = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
const calls = JSON.parse(readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(calls.map(([fn, ...args]) => mod[fn](...args))));
"""


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in KEYS:
        monkeypatch.setenv(name, 'test')


def js(*calls):
    if NODE is None:
        pytest.skip('node 不可用，跳过 render.js 测试')
    done = subprocess.run([NODE, '--input-type=module', '-e', HARNESS, str(STATIC / 'render.js')],
                          input=json.dumps(calls), capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def cite(answer, n, state, nth=0):
    # 与后处理同一个换算函数，验证 JS 侧按 UTF-16 下标切片。
    pos = -1
    for _ in range(nth + 1):
        pos = answer.index(f'[{n}]', pos + 1)
    return {'n': n, 'start': utf16_offset(answer, pos), 'end': utf16_offset(answer, pos + len(f'[{n}]')), 'state': state}


def test_demo_page_and_static_assets_served_with_csp(tmp_path, embed_url):
    cfg = config(tmp_path)
    with api_client(cfg, deps(cfg, embed_url)) as client:
        page = client.get('/')
        assert page.status_code == 200 and page.headers['content-type'].startswith('text/html')
        csp = page.headers['content-security-policy']
        assert "script-src 'self'" in csp and "default-src 'none'" in csp and 'unsafe-inline' not in csp
        assert '/static/app.js' in page.text
        for name in ('app.js', 'render.js'):
            asset = client.get(f'/static/{name}')
            assert asset.status_code == 200 and 'javascript' in asset.headers['content-type']
        assert client.get('/static/style.css').headers['content-type'].startswith('text/css')
        # 浏览器默认会请求 favicon；页面声明的同源图标要真能取到，不然控制台报 404。
        assert '/static/favicon.svg' in page.text
        assert client.get('/static/favicon.svg').headers['content-type'].startswith('image/svg+xml')
        assert client.get('/static/%2e%2e/app.py').status_code == 404
        assert client.get('/static/nope.js').status_code == 404


def test_render_escapes_raw_html_and_unsafe_links():
    html, links = js(['renderAnswer', '<script>alert(1)</script> **粗** <img src=x onerror=alert(1)>'],
                     ['renderAnswer', '[好](https://example.com/a?b=1&c=2) [坏](javascript:alert(1))'])
    assert '<script' not in html and '<img' not in html
    assert '&lt;script&gt;' in html and '<strong>粗</strong>' in html
    assert 'href="https://example.com/a?b=1&amp;c=2"' in links and 'rel="noopener noreferrer"' in links
    assert 'href="javascript' not in links


def test_render_citations_by_utf16_offsets_states_and_supplement():
    answer = '部署先装依赖😀[1]。再改配置[2]，见[9]。\n\n**补充说明（以下内容不来自资料）**\n常识[3]。'
    final = {'citations': [cite(answer, 1, 'ok'), cite(answer, 2, 'unsupported'),
                           cite(answer, 9, 'invalid_number'), cite(answer, 3, 'in_supplement')],
             'supplement': {'start': utf16_offset(answer, answer.index('**补充'))},
             'sources': [{'cite': 1}, {'cite': 2}, {'cite': 3}]}
    html, = js(['renderAnswer', answer, final, 't1'])
    assert '<a class="cite" href="#t1-src-1">[1]</a>' in html and '😀' in html
    assert 'class="cite cite-grey" href="#t1-src-2" title="未能在该资料中核实"' in html
    assert '<span class="cite cite-grey" title="引用编号无效">[9]</span>' in html
    body, supplement = html.split('<div class="supplement"')
    assert '补充说明不应引用资料' in supplement and '[3]' not in body
    assert all(html.count(f'[{n}]') == 1 for n in (1, 2, 3, 9))


def test_render_code_protection_mismatch_streaming_and_private_marks():
    code = '用法：`arr[1]`，示例：\n```python\nx = arr[2]\n```\n结论[1]。'
    marked = 'a\ue000b结论[1]'
    code_html, mismatch, streaming, private = js(
        ['renderAnswer', code, {'citations': [cite(code, 1, 'ok', nth=1)], 'sources': [{'cite': 1}]}],
        ['renderAnswer', '没有引用的句子。', {'citations': [{'n': 1, 'start': 0, 'end': 3, 'state': 'ok'}]}],
        ['renderAnswer', '结论[1]'],
        ['renderAnswer', marked, {'citations': [cite(marked, 1, 'ok')], 'sources': [{'cite': 1}]}])
    assert '<code>arr[1]</code>' in code_html and '<pre><code>x = arr[2]</code></pre>' in code_html
    assert code_html.count('<sup>') == 1
    assert '<sup>' not in mismatch and '没有引用的句子' in mismatch
    assert '<sup>' not in streaming and '结论[1]' in streaming
    assert '\ue000' not in private and '<sup>' in private and 'ab结论' in private


def test_render_markdown_blocks_sources_and_sse_frames():
    md = '## 标题\n- 一\n- 二\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n1. x\n2. y'
    sources = [{'cite': 1, 'title': '<b>手册</b>', 'section': '第2章 > 2.3', 'pages': '12-13', 'doc_id': '0123456789abcdef',
                'images': [{'url': '/api/images/0123456789abcdef/v1/a.png', 'caption': '<图>'},
                           {'url': 'javascript:alert(1)', 'caption': 'x'}, {'url': None, 'caption': 'y'}]},
               {'cite': 2, 'title': 't', 'doc_id': 'bad', 'images': []}]
    frames = 'data: {"type":"token","text":"a\\n"}\n\ndata: {"type":"fi'
    blocks, items, sse = js(['renderAnswer', md], ['renderSources', sources, 't1'], ['parseSSE', frames])
    assert '<h4>标题</h4>' in blocks and '<ul><li>一</li><li>二</li></ul>' in blocks
    assert '<th scope="col">a</th>' in blocks and '<td>1</td>' in blocks and '<ol><li>x</li><li>y</li></ol>' in blocks
    assert '&lt;b&gt;手册&lt;/b&gt;' in items and 'id="t1-src-1"' in items and '第 12-13 页' in items
    assert items.count('href="/api/docs/0123456789abcdef/file?inline=true" target="_blank"') == 1
    assert items.count('href="/api/docs/0123456789abcdef/file" download') == 1 and 'alt="&lt;图&gt;"' in items
    assert items.count('<img') == 1 and 'javascript' not in items
    assert sse == {'events': [{'type': 'token', 'text': 'a\n'}], 'rest': 'data: {"type":"fi'}


def test_render_attachments_scope_label_and_doc_id():
    attachments = [{'url': '/api/images/0123456789abcdef/v1/a.png', 'caption': '<架构图>'},
                   {'url': 'https://evil.example/x.png', 'caption': 'x'}, {'url': None}]
    html, empty, label, none, ok, bad = js(['renderAttachments', attachments], ['renderAttachments', None],
        ['scopeLabel', {'file_types': ['pdf', 'docx'], 'doc_ids': ['0123456789abcdef', 'fedcba9876543210']},
         {'0123456789abcdef': '<手册>'}],
        ['scopeLabel', None], ['isDocId', '0123456789abcdef'], ['isDocId', '../etc'])
    assert html.count('<img') == 1 and 'evil' not in html and '&lt;架构图&gt;' in html and '<figcaption>' in html
    assert empty == '' and none == '' and ok is True and bad is False
    # 纯文本，由 textContent 写入，不在这里转义。
    assert label == '检索范围：pdf/docx 文件、《<手册>》、《fedcba9876543210》'
