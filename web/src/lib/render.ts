// 纯函数：安全 Markdown 子集 + 按 final.citations 的 UTF-16 位置渲染引用（后处理 §8/§11）。
// 不依赖 DOM，vitest 直接调用。输出 HTML 字符串：所有文本先转义，只有本文件拼出的标签是真标签。
import hljs from 'highlight.js/lib/common';
import katex from 'katex';
import type { Citation, Final, ImageRef, Source } from './types';

export const STAGES: Record<string, string> = {
  preprocessing: '理解问题', retrieving: '检索资料', reranking: '重排',
  rewriting: '没找到足够的资料，换个说法再检索', building_context: '整理资料', generating: '生成回答',
};

// 后处理 §4：这三种标灰并给说明；ok / unverified 正常显示。
export const GREY: Record<string, string> = {
  unsupported: '未能在该资料中核实',
  invalid_number: '引用编号无效',
  in_supplement: '补充说明不应引用资料',
};

const DOC_ID = /^[0-9a-f]{16}$/;
export const IMAGE_URL = /^\/api\/images\/[0-9a-f]{16}\/v[1-9]\d*\/[^/?#]+$/;
const CITE_TEXT = /^\[\d{1,3}\]$/;
// 私有区字符做占位：引用 U+E000+i、行内代码 U+E800+i、行内公式 U+F000+i。原文里的同区字符先删掉。
const range = (lo: number, hi: number) => new RegExp(`[${String.fromCharCode(lo)}-${String.fromCharCode(hi)}]`, 'g');
const MARKS = range(0xe000, 0xf8ff);
const CITE_MARK = range(0xe000, 0xe7ff);
const CODE_MARK = range(0xe800, 0xefff);
const MATH_MARK = range(0xf000, 0xf7ff);

export function escapeHtml(s: unknown): string {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}

const strip = (s: string) => s.replace(MARKS, '');

export function isDocId(value: unknown): value is string {
  return DOC_ID.test(String(value ?? ''));
}

export function isImageUrl(value: unknown): value is string {
  return IMAGE_URL.test(String(value ?? ''));
}

// KaTeX 只输出 MathML（D-020）：没有内联样式，CSP 不放宽。解析失败按原文转义显示。
export function renderMath(tex: string, display: boolean): string {
  try {
    const html = katex.renderToString(tex, { output: 'mathml', displayMode: display, throwOnError: true, trust: false, strict: 'ignore' });
    return display ? `<div class="math-block">${html}</div>` : html;
  } catch {
    const src = escapeHtml(display ? `$$${tex}$$` : `$${tex}$`);
    return display ? `<pre class="math-error">${src}</pre>` : `<code class="math-error">${src}</code>`;
  }
}

// 代码高亮：hljs 输出只有 span.hljs-* 且已转义；未知语言按纯文本转义。
export function renderCode(code: string, lang: string): string {
  const name = lang.trim().split(/\s+/)[0]?.toLowerCase() ?? '';
  let body: string;
  if (name && hljs.getLanguage(name)) body = hljs.highlight(code, { language: name, ignoreIllegals: true }).value;
  else body = escapeHtml(code);
  const label = escapeHtml(name || 'text');
  return `<div class="code"><div class="code-head"><span>${label}</span>`
    + `<button type="button" class="copy-code" aria-label="复制代码">复制</button></div>`
    + `<pre><code class="hljs">${body}</code></pre></div>`;
}

interface Ctx {
  list: Citation[];
  sources: Set<number>;
  prefix: string;
  // 原文模式：图片相对路径改写到 /api/images/{doc}/v{n}/，标题带 data-path 供定位。
  doc?: { docId: string; version: number };
}

// 把 [start,end) 内的引用换成占位符；位置对不上 `[n]` 的引用保持原文，不猜。
function markCitations(text: string, citations: Citation[], start: number, end: number) {
  const list: Citation[] = [];
  let out = '';
  let pos = start;
  const sorted = citations
    .filter((c) => Number.isInteger(c.start) && Number.isInteger(c.end) && c.start >= start && c.end <= end)
    .sort((a, b) => a.start - b.start);
  for (const c of sorted) {
    if (c.start < pos || !CITE_TEXT.test(text.slice(c.start, c.end))) continue;
    out += strip(text.slice(pos, c.start)) + String.fromCharCode(0xe000 + list.length);
    list.push(c);
    pos = c.end;
  }
  return { text: out + strip(text.slice(pos, end)), list };
}

function citeHtml(c: Citation, ctx: Ctx): string {
  const n = Number(c.n);
  const hint = GREY[c.state];
  const cls = hint ? 'cite cite-grey' : 'cite';
  const data = ` data-cite="${n}" data-turn="${escapeHtml(ctx.prefix)}"${hint ? ` data-hint="${escapeHtml(hint)}"` : ''}`;
  const sr = hint ? `<span class="sr-only">（${escapeHtml(hint)}）</span>` : '';
  if (ctx.sources.has(n)) {
    return `<sup><a class="${cls}" href="#${escapeHtml(ctx.prefix)}-src-${n}"${data}>${n}</a>${sr}</sup>`;
  }
  return `<sup><span class="${cls}"${data}>${n}</span>${sr}</sup>`;
}

// 行内公式：$x$，开头结尾不能是空白，结尾 $ 后不能紧跟数字（避免“$5 和 $10”误判）。
const INLINE_MATH = /\$([^\s$](?:[^$\n]*[^\s$])?)\$(?!\d)/g;
const IMAGE = /!\[((?:\\\]|[^\]\n])*)\]\(images\/([^\s()/?#]+)\)/g;

function inline(s: string, ctx: Ctx): string {
  const codes: string[] = [];
  const maths: string[] = [];
  let h = s.replace(/`([^`\n]+)`/g, (_, code: string) => String.fromCharCode(0xe800 + codes.push(code) - 1));
  h = h.replace(INLINE_MATH, (_, tex: string) => String.fromCharCode(0xf000 + maths.push(tex) - 1));
  h = escapeHtml(h);
  if (ctx.doc) {
    const { docId, version } = ctx.doc;
    h = h.replace(IMAGE, (_, alt: string, file: string) => {
      const url = `/api/images/${docId}/v${version}/${file}`;
      return `<img src="${url}" alt="${alt.replace(/\\\]/g, ']')}" loading="lazy">`;
    });
  }
  h = h.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
    (_, label: string, url: string) => `<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`);
  h = h.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, '$1<em>$2</em>');
  h = h.replace(CODE_MARK, (m) => `<code>${escapeHtml(codes[m.charCodeAt(0) - 0xe800] ?? '')}</code>`);
  h = h.replace(MATH_MARK, (m) => renderMath(strip(maths[m.charCodeAt(0) - 0xf000] ?? ''), false));
  return h.replace(CITE_MARK, (m) => {
    const c = ctx.list[m.charCodeAt(0) - 0xe000];
    return c ? citeHtml(c, ctx) : '';
  });
}

const FENCE = /^[ \t]*(`{3,}|~{3,})(.*)$/;
const HEADING = /^(#{1,6})\s+(.*)$/;
const LIST_ITEM = /^\s*([-*+]|\d+[.)])\s+(.*)$/;
const RULE = /^\s*([-*_])(\s*\1){2,}\s*$/;
const TABLE_SEP = /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/;
const MATH_FENCE = /^\s*\$\$\s*$/;
const MATH_LINE = /^\s*\$\$(.+)\$\$\s*$/;
const cells = (line: string) => line.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());

function blocks(src: string, ctx: Ctx): string {
  const lines = src.split('\n');
  const out: string[] = [];
  const path: string[] = [];
  let para: string[] = [];
  const flush = () => {
    if (para.length) out.push(`<p>${para.map((l) => inline(l, ctx)).join('<br>')}</p>`);
    para = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const fence = FENCE.exec(line);
    if (fence) {
      // 代码块原样转义；未闭合时一直到结尾，和后处理 mask_code 一致。
      flush();
      const close = new RegExp(`^[ \\t]*${fence[1]![0] === '`' ? '`' : '~'}{${fence[1]!.length},}\\s*$`);
      const body: string[] = [];
      for (i++; i < lines.length && !close.test(lines[i]!); i++) body.push(lines[i]!);
      out.push(renderCode(strip(body.join('\n')), fence[2] ?? ''));
      continue;
    }
    const one = MATH_LINE.exec(line);
    if (one) {
      flush();
      out.push(renderMath(strip(one[1]!), true));
      continue;
    }
    if (MATH_FENCE.test(line)) {
      // 未闭合（流式阶段常见）时不吞后文，按普通段落显示。
      const end = lines.findIndex((l, j) => j > i && MATH_FENCE.test(l));
      if (end > i) {
        flush();
        out.push(renderMath(strip(lines.slice(i + 1, end).join('\n')), true));
        i = end;
        continue;
      }
    }
    const heading = HEADING.exec(line);
    if (heading) {
      flush();
      const depth = heading[1]!.length;
      if (ctx.doc) {
        // 原文：保留级别，data-path 是“祖先 > 本级”的标题路径，和 sources.section 对齐。
        path.length = depth - 1;
        path[depth - 1] = strip(heading[2]!).trim();
        const full = path.filter(Boolean).join(' > ');
        out.push(`<h${depth} data-path="${escapeHtml(full)}">${inline(heading[2]!, ctx)}</h${depth}>`);
      } else {
        const level = Math.min(depth + 2, 6);
        out.push(`<h${level}>${inline(heading[2]!, ctx)}</h${level}>`);
      }
      continue;
    }
    if (line.includes('|') && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1]!)) {
      flush();
      const head = cells(line).map((c) => `<th scope="col">${inline(c, ctx)}</th>`).join('');
      const rows: string[] = [];
      for (i += 2; i < lines.length && lines[i]!.includes('|') && lines[i]!.trim(); i++) {
        rows.push(`<tr>${cells(lines[i]!).map((c) => `<td>${inline(c, ctx)}</td>`).join('')}</tr>`);
      }
      i--;
      out.push(`<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`);
      continue;
    }
    if (RULE.test(line)) {
      flush();
      out.push('<hr>');
      continue;
    }
    if (LIST_ITEM.test(line)) {
      flush();
      const ordered = /\d/.test(LIST_ITEM.exec(line)![1]!);
      const items: string[] = [];
      for (; i < lines.length; i++) {
        const m = LIST_ITEM.exec(lines[i]!);
        if (m && /\d/.test(m[1]!) === ordered) items.push(inline(m[2]!, ctx));
        else if (lines[i]!.trim() && /^\s{2,}/.test(lines[i]!) && items.length) items[items.length - 1] += '<br>' + inline(lines[i]!.trim(), ctx);
        else break;
      }
      i--;
      const tag = ordered ? 'ol' : 'ul';
      out.push(`<${tag}>${items.map((x) => `<li>${x}</li>`).join('')}</${tag}>`);
      continue;
    }
    if (/^\s*>/.test(line)) {
      flush();
      const quote: string[] = [];
      for (; i < lines.length && /^\s*>/.test(lines[i]!); i++) quote.push(inline(lines[i]!.replace(/^\s*>\s?/, ''), ctx));
      i--;
      out.push(`<blockquote><p>${quote.join('<br>')}</p></blockquote>`);
      continue;
    }
    if (!line.trim()) flush();
    else para.push(line);
  }
  flush();
  return out.join('\n');
}

// final 为空时是流式阶段：照样渲染 Markdown，但 [n] 只当普通文字（后处理 §11）。
export function renderAnswer(answer: string, final: Final | null = null, prefix = 'm'): string {
  const text = String(answer ?? '');
  const citations = Array.isArray(final?.citations) ? final.citations : [];
  const sources = new Set((final?.sources ?? []).map((s) => Number(s.cite)));
  const part = (start: number, end: number) => {
    const { text: marked, list } = markCitations(text, citations, start, end);
    return blocks(marked, { list, sources, prefix });
  };
  const supp = final?.supplement?.start;
  if (Number.isInteger(supp) && supp! > 0 && supp! < text.length) {
    return `${part(0, supp!)}\n<div class="supplement" role="note" aria-label="补充说明，不来自资料">${part(supp!, text.length)}</div>`;
  }
  return part(0, text.length);
}

// 原文 full.md：同一个渲染器，图片改写到 /api/images，标题带 data-path。调用方保证 docId 合法。
export function renderDoc(markdown: string, docId: string, version: number): string {
  if (!isDocId(docId) || !Number.isInteger(version) || version < 1) return '';
  return blocks(strip(String(markdown ?? '')), { list: [], sources: new Set(), prefix: 'doc', doc: { docId, version } });
}

// sources.section 形如“第2章 部署 > 2.3 配置”：先整路径匹配，再退到最后一级标题。
export function findSection(paths: string[], section: string): number {
  const want = section.split('>').map((s) => s.trim()).filter(Boolean);
  if (!want.length) return -1;
  const joined = want.join(' > ');
  const exact = paths.findIndex((p) => p === joined);
  if (exact >= 0) return exact;
  const last = want[want.length - 1]!;
  return paths.findIndex((p) => p.split(' > ').at(-1) === last);
}

export function safeImages(images: ImageRef[] | null | undefined): ImageRef[] {
  return (images ?? []).filter((img) => isImageUrl(img?.url));
}

export function sourceMeta(s: Source): string {
  return [s.section, s.pages ? `第 ${s.pages} 页` : ''].filter(Boolean).join(' · ');
}

// 硬过滤的可读说明（纯文本，React 负责转义）：文件类型 + 文档标题。
export function scopeLabel(filters: { file_types?: string[]; doc_ids?: string[] } | null | undefined,
  titles: Record<string, string> = {}): string {
  const parts: string[] = [];
  if (filters?.file_types?.length) parts.push(filters.file_types.join('/') + ' 文件');
  for (const id of filters?.doc_ids ?? []) parts.push(`《${titles[id] ?? id}》`);
  return parts.length ? '检索范围：' + parts.join('、') : '';
}
