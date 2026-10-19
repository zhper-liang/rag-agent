// 迁自 tests/api/test_demo.py 的渲染器用例，外加公式、高亮、原文模式。
import { describe, expect, it } from 'vitest';
import { findSection, isDocId, renderAnswer, renderDoc, safeImages, scopeLabel, sourceMeta } from './render';
import { parseSSE } from './sse';
import type { Citation } from './types';

// JS 字符串下标本身就是 UTF-16 单元，和后处理 utf16_offset 一致。
function cite(answer: string, n: number, state: string, nth = 0): Citation {
  let pos = -1;
  for (let k = 0; k <= nth; k++) pos = answer.indexOf(`[${n}]`, pos + 1);
  return { n, start: pos, end: pos + `[${n}]`.length, state };
}

describe('转义', () => {
  it('原始 HTML 和不安全链接都不会变成标签', () => {
    const html = renderAnswer('<script>alert(1)</script> **粗** <img src=x onerror=alert(1)>');
    expect(html).not.toContain('<script');
    expect(html).not.toContain('<img');
    expect(html).toContain('&lt;script&gt;');
    expect(html).toContain('<strong>粗</strong>');
    const links = renderAnswer('[好](https://example.com/a?b=1&c=2) [坏](javascript:alert(1))');
    expect(links).toContain('href="https://example.com/a?b=1&amp;c=2"');
    expect(links).toContain('rel="noopener noreferrer"');
    expect(links).not.toContain('href="javascript');
  });
});

describe('引用', () => {
  it('按 UTF-16 偏移、状态和补充说明渲染', () => {
    const answer = '部署先装依赖😀[1]。再改配置[2]，见[9]。\n\n**补充说明（以下内容不来自资料）**\n常识[3]。';
    const final = {
      citations: [cite(answer, 1, 'ok'), cite(answer, 2, 'unsupported'), cite(answer, 9, 'invalid_number'), cite(answer, 3, 'in_supplement')],
      supplement: { start: answer.indexOf('**补充') },
      sources: [{ cite: 1 }, { cite: 2 }, { cite: 3 }],
    };
    const html = renderAnswer(answer, final, 't1');
    expect(html).toContain('<a class="cite" href="#t1-src-1" data-cite="1" data-turn="t1">1</a>');
    expect(html).toContain('😀');
    expect(html).toContain('class="cite cite-grey" href="#t1-src-2" data-cite="2" data-turn="t1" data-hint="未能在该资料中核实"');
    expect(html).toContain('<span class="cite cite-grey" data-cite="9" data-turn="t1" data-hint="引用编号无效">9</span>');
    const [body, supplement] = html.split('<div class="supplement"');
    expect(supplement).toContain('补充说明不应引用资料');
    expect(body).not.toContain('data-cite="3"');
    for (const n of [1, 2, 3, 9]) expect(html.split(`data-cite="${n}"`).length).toBe(2);
    expect(html).not.toMatch(/\[\d\]/);
  });

  it('代码里的 [n] 不动、位置对不上不猜、流式阶段不渲染、私有区字符删掉', () => {
    const code = '用法：`arr[1]`，示例：\n```python\nx = arr[2]\n```\n结论[1]。';
    const codeHtml = renderAnswer(code, { citations: [cite(code, 1, 'ok', 1)], sources: [{ cite: 1 }] });
    expect(codeHtml).toContain('<code>arr[1]</code>');
    expect(codeHtml).toContain('arr[');
    expect(codeHtml.split('<sup>').length).toBe(2);
    const mismatch = renderAnswer('没有引用的句子。', { citations: [{ n: 1, start: 0, end: 3, state: 'ok' }] });
    expect(mismatch).not.toContain('<sup>');
    expect(mismatch).toContain('没有引用的句子');
    const streaming = renderAnswer('结论[1]');
    expect(streaming).not.toContain('<sup>');
    expect(streaming).toContain('结论[1]');
    const marked = 'a\ue000b结论[1]';
    const priv = renderAnswer(marked, { citations: [cite(marked, 1, 'ok')], sources: [{ cite: 1 }] });
    expect(priv).not.toContain('\ue000');
    expect(priv).toContain('<sup>');
    expect(priv).toContain('ab结论');
  });
});

describe('Markdown 块', () => {
  it('标题、列表、表格、引用块', () => {
    const html = renderAnswer('## 标题\n- 一\n- 二\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n1. x\n2. y\n\n> 引');
    expect(html).toContain('<h4>标题</h4>');
    expect(html).toContain('<ul><li>一</li><li>二</li></ul>');
    expect(html).toContain('<th scope="col">a</th>');
    expect(html).toContain('<td>1</td>');
    expect(html).toContain('<ol><li>x</li><li>y</li></ol>');
    expect(html).toContain('<blockquote><p>引</p></blockquote>');
  });

  it('代码块高亮且已转义，未知语言按纯文本', () => {
    const html = renderAnswer('```python\nx = 1 < 2\n```\n```nope\n<b>\n```');
    expect(html).toContain('<span class="hljs-number">1</span>');
    expect(html).toContain('&lt;');
    expect(html).toContain('<span>nope</span>');
    expect(html).toContain('<code class="hljs">&lt;b&gt;</code>');
    expect(html).not.toContain('style=');
  });

  it('未闭合代码块一直到结尾', () => {
    const html = renderAnswer('前\n```\nlet a = 1;\n后[1]');
    expect(html).toContain('后[1]');
    expect(html.indexOf('后[1]')).toBeGreaterThan(html.indexOf('<pre>'));
  });
});

describe('公式', () => {
  it('KaTeX 只输出 MathML，没有内联样式', () => {
    const html = renderAnswer('勾股 $a^2+b^2=c^2$ 成立。\n\n$$\n\\frac{1}{2}\n$$\n\n$$\\sqrt{x}$$');
    expect(html.split('<math').length).toBe(4);
    expect(html).toContain('display="block"');
    expect(html).not.toContain('style=');
  });

  it('金额不当公式，坏公式按原文转义', () => {
    const money = renderAnswer('价格 $5 和 $10 之间');
    expect(money).not.toContain('<math');
    const bad = renderAnswer('坏 $\\frac{<b>$ 了');
    expect(bad).toContain('class="math-error"');
    expect(bad).not.toContain('<b>');
    expect(bad).not.toContain('style=');
  });

  it('未闭合的 $$ 不吞后文', () => {
    const html = renderAnswer('$$\n\\frac{1}{2}\n后面的话');
    expect(html).not.toContain('<math');
    expect(html).toContain('后面的话');
  });

  it('公式里的 [n] 引用占位被删掉，不会混进 TeX', () => {
    const text = '式 $x_[1]$ 与结论[1]';
    const html = renderAnswer(text, { citations: [cite(text, 1, 'ok', 1)], sources: [{ cite: 1 }] });
    expect(html).not.toMatch(/[\ue000-\uf8ff]/);
    expect(html.split('<sup>').length).toBe(2);
  });
});

describe('原文模式', () => {
  it('图片改写到 /api/images，标题带路径，其余照样转义', () => {
    const md = '# 第2章 部署\n## 2.3 配置\n![架构\\]图](images/a.png) ![x](../etc/passwd) ![y](https://evil/x.png)\n<script>';
    const html = renderDoc(md, '0123456789abcdef', 2);
    expect(html).toContain('<h1 data-path="第2章 部署">');
    expect(html).toContain('<h2 data-path="第2章 部署 &gt; 2.3 配置">');
    expect(html).toContain('<img src="/api/images/0123456789abcdef/v2/a.png" alt="架构]图" loading="lazy">');
    expect(html.split('<img').length).toBe(2);
    expect(html).toContain('&lt;script&gt;');
    expect(renderDoc(md, '../x', 1)).toBe('');
  });

  it('按 section 定位：整路径优先，退到最后一级', () => {
    const paths = ['第2章 部署', '第2章 部署 > 2.3 配置', '附录 > 2.3 配置'];
    expect(findSection(paths, '第2章 部署 > 2.3 配置')).toBe(1);
    expect(findSection(paths, '其他 > 2.3 配置')).toBe(1);
    expect(findSection(paths, '不存在')).toBe(-1);
    expect(findSection(paths, '')).toBe(-1);
  });
});

describe('其它', () => {
  it('来源图片过滤、元信息、范围说明、doc_id、SSE 帧', () => {
    const imgs = safeImages([{ url: '/api/images/0123456789abcdef/v1/a.png' }, { url: 'javascript:alert(1)' }, { url: null as unknown as string }]);
    expect(imgs).toHaveLength(1);
    expect(sourceMeta({ cite: 1, section: '第2章 > 2.3', pages: '12-13' })).toBe('第2章 > 2.3 · 第 12-13 页');
    expect(scopeLabel({ file_types: ['pdf', 'docx'], doc_ids: ['0123456789abcdef', 'fedcba9876543210'] }, { '0123456789abcdef': '<手册>' }))
      .toBe('检索范围：pdf/docx 文件、《<手册>》、《fedcba9876543210》');
    expect(scopeLabel(null)).toBe('');
    expect(isDocId('0123456789abcdef')).toBe(true);
    expect(isDocId('../etc')).toBe(false);
    expect(parseSSE('data: {"type":"token","text":"a\\n"}\n\ndata: bad\n\ndata: {"type":"fi'))
      .toEqual({ events: [{ type: 'token', text: 'a\n' }], rest: 'data: {"type":"fi' });
  });
});
