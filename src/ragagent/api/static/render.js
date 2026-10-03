// 纯函数：安全 Markdown 子集 + 按 final.citations 的 UTF-16 位置渲染引用（后处理 §8/§11）。
// 不依赖 DOM，tests/api/test_demo.py 用 node 直接调用。

export const STAGES = {
  preprocessing: '理解问题', retrieving: '检索资料', reranking: '重排',
  rewriting: '没找到足够的资料，换个说法再检索', building_context: '整理资料', generating: '生成回答',
};

// 后处理 §4：这三种标灰并给悬停说明；ok / unverified 正常显示。
const GREY = {
  unsupported: '未能在该资料中核实',
  invalid_number: '引用编号无效',
  in_supplement: '补充说明不应引用资料',
};

const DOC_ID = /^[0-9a-f]{16}$/;
const IMAGE_URL = /^\/api\/images\/[0-9a-f]{16}\/v[1-9]\d*\/[^/?#]+$/;
const CITE_TEXT = /^\[\d{1,3}\]$/;
// 私有区字符做占位：引用 U+E000+i、行内代码 U+E800+i。原文里的同区字符先删掉。
const range = (lo, hi) => new RegExp(`[${String.fromCharCode(lo)}-${String.fromCharCode(hi)}]`, 'g');
const MARKS = range(0xe000, 0xf8ff);
const CITE_MARK = range(0xe000, 0xe7ff);
const CODE_MARK = range(0xe800, 0xefff);

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

const strip = (s) => s.replace(MARKS, '');

// 把 [start,end) 内的引用换成占位符；位置对不上 `[n]` 的引用保持原文，不猜。
function markCitations(text, citations, start, end) {
  const list = [];
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

function citeHtml(c, ctx) {
  const n = Number(c.n);
  const hint = GREY[c.state];
  const label = `[${n}]`;
  const cls = hint ? 'cite cite-grey' : 'cite';
  const title = hint ? ` title="${escapeHtml(hint)}"` : '';
  const sr = hint ? `<span class="sr-only">（${escapeHtml(hint)}）</span>` : '';
  if (ctx.sources.has(n)) {
    return `<sup><a class="${cls}" href="#${ctx.prefix}-src-${n}"${title}>${label}</a>${sr}</sup>`;
  }
  return `<sup><span class="${cls}"${title}>${label}</span>${sr}</sup>`;
}

function inline(s, ctx) {
  const codes = [];
  let h = s.replace(/`([^`\n]+)`/g, (_, code) => String.fromCharCode(0xe800 + codes.push(code) - 1));
  h = escapeHtml(h);
  h = h.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
    (_, label, url) => `<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`);
  h = h.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, '$1<em>$2</em>');
  h = h.replace(CODE_MARK, (m) => `<code>${escapeHtml(codes[m.charCodeAt(0) - 0xe800] ?? '')}</code>`);
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
const cells = (line) => line.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());

function blocks(src, ctx) {
  const lines = src.split('\n');
  const out = [];
  let para = [];
  const flush = () => {
    if (para.length) out.push(`<p>${para.map((l) => inline(l, ctx)).join('<br>')}</p>`);
    para = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fence = FENCE.exec(line);
    if (fence) {
      // 代码块原样转义；未闭合时一直到结尾，和后处理 mask_code 一致。
      flush();
      const close = new RegExp(`^[ \\t]*${fence[1][0] === '`' ? '`' : '~'}{${fence[1].length},}\\s*$`);
      const body = [];
      for (i++; i < lines.length && !close.test(lines[i]); i++) body.push(lines[i]);
      out.push(`<pre><code>${escapeHtml(strip(body.join('\n')))}</code></pre>`);
      continue;
    }
    const heading = HEADING.exec(line);
    if (heading) {
      flush();
      const level = Math.min(heading[1].length + 2, 6);
      out.push(`<h${level}>${inline(heading[2], ctx)}</h${level}>`);
      continue;
    }
    if (line.includes('|') && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1])) {
      flush();
      const head = cells(line).map((c) => `<th scope="col">${inline(c, ctx)}</th>`).join('');
      const rows = [];
      for (i += 2; i < lines.length && lines[i].includes('|') && lines[i].trim(); i++) {
        rows.push(`<tr>${cells(lines[i]).map((c) => `<td>${inline(c, ctx)}</td>`).join('')}</tr>`);
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
      const ordered = /\d/.test(LIST_ITEM.exec(line)[1]);
      const items = [];
      for (; i < lines.length; i++) {
        const m = LIST_ITEM.exec(lines[i]);
        if (m && /\d/.test(m[1]) === ordered) items.push(inline(m[2], ctx));
        else if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && items.length) items[items.length - 1] += '<br>' + inline(lines[i].trim(), ctx);
        else break;
      }
      i--;
      const tag = ordered ? 'ol' : 'ul';
      out.push(`<${tag}>${items.map((x) => `<li>${x}</li>`).join('')}</${tag}>`);
      continue;
    }
    if (/^\s*>/.test(line)) {
      flush();
      const quote = [];
      for (; i < lines.length && /^\s*>/.test(lines[i]); i++) quote.push(inline(lines[i].replace(/^\s*>\s?/, ''), ctx));
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

// final 为空时是流式阶段：[n] 按普通文本显示（后处理 §11）。
export function renderAnswer(answer, final = null, prefix = 'm') {
  const text = String(answer ?? '');
  const citations = Array.isArray(final?.citations) ? final.citations : [];
  const sources = new Set((final?.sources ?? []).map((s) => Number(s.cite)));
  const part = (start, end) => {
    const { text: marked, list } = markCitations(text, citations, start, end);
    return blocks(marked, { list, sources, prefix });
  };
  const supp = final?.supplement?.start;
  if (Number.isInteger(supp) && supp > 0 && supp < text.length) {
    return `${part(0, supp)}\n<div class="supplement" role="note" aria-label="补充说明，不来自资料">${part(supp, text.length)}</div>`;
  }
  return part(0, text.length);
}

export function renderSources(sources, prefix = 'm') {
  return (sources ?? []).map((s) => {
    const n = Number(s.cite);
    const meta = [s.section, s.pages ? `第 ${s.pages} 页` : ''].filter(Boolean).map(escapeHtml).join(' · ');
    const link = DOC_ID.test(String(s.doc_id ?? ''))
      ? ` <a class="doc-link" href="/api/docs/${s.doc_id}/file" download>下载原文</a>` : '';
    const images = (s.images ?? []).filter((img) => IMAGE_URL.test(String(img.url ?? ''))).map((img) => {
      const alt = escapeHtml(img.caption || `来源 ${n} 的图片`);
      return `<a href="${escapeHtml(img.url)}" target="_blank" rel="noopener"><img src="${escapeHtml(img.url)}" alt="${alt}" loading="lazy"></a>`;
    }).join('');
    return `<li id="${prefix}-src-${n}"><span class="src-n">[${n}]</span> <span class="src-title">${escapeHtml(s.title ?? '')}</span>`
      + `${meta ? `<span class="src-meta">${meta}</span>` : ''}${link}${images ? `<div class="src-images">${images}</div>` : ''}</li>`;
  }).join('');
}

// SSE 帧切分：返回完整帧解析出的事件和剩余缓冲；只认 data 行（sse.py 每帧一行 JSON）。
export function parseSSE(buffer) {
  const events = [];
  const frames = buffer.split('\n\n');
  const rest = frames.pop();
  for (const frame of frames) {
    const data = frame.split('\n').filter((l) => l.startsWith('data:')).map((l) => l.slice(5).trimStart()).join('\n');
    if (!data) continue;
    try {
      events.push(JSON.parse(data));
    } catch {
      // 坏帧丢弃，不中断整条流。
    }
  }
  return { events, rest };
}
