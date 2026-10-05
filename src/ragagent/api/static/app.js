// 演示页 DOM 层：POST /api/chat 读 SSE，final 后重渲染引用；反馈走 /api/feedback。
// 侧栏：会话历史只存 localStorage；检索范围作为硬过滤 filters 发给接口；文档列表来自 GET /api/docs。
import { STAGES, isDocId, parseSSE, renderAnswer, renderAttachments, renderSources, scopeLabel } from './render.js';

const $ = (sel) => document.querySelector(sel);
const log = $('#log');
const form = $('#ask');
const input = $('#question');
const sendBtn = $('#send');
const stopBtn = $('#stop');
const status = $('#status');

const STORE = 'ragagent.sessions.v1';
// 与 retrieval/filters.py 的 FILE_TYPES、MAX_FILTER_ITEMS 一致。
const TYPE_GROUPS = [
  ['PDF', ['pdf']], ['Word', ['doc', 'docx']], ['PPT', ['ppt', 'pptx']],
  ['图片', ['png', 'jpg', 'jpeg']], ['文本', ['md', 'txt']],
];
const MAX_DOCS = 100;
const narrow = window.matchMedia('(max-width: 760px)');

const newSessionId = () => {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return 'web-' + Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
};

let sessionId = newSessionId();
let controller = null;
let turnSeq = 0;
let sessions = loadSessions();
const picked = new Map(); // doc_id -> title
let shownDocs = { docs: [], truncated: false, q: '' };
let docSeq = 0;

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function button(cls, text, label) {
  const node = el('button', cls, text);
  node.type = 'button';
  if (label) node.setAttribute('aria-label', label);
  return node;
}

function nearBottom() {
  return log.scrollHeight - log.scrollTop - log.clientHeight < 80;
}

function scrollDown(force = false) {
  if (force || nearBottom()) log.scrollTop = log.scrollHeight;
}

function announce(text) {
  status.textContent = text;
}

function setBusy(busy) {
  sendBtn.disabled = busy;
  stopBtn.hidden = !busy;
  input.setAttribute('aria-busy', String(busy));
  log.querySelectorAll('.unscope').forEach((b) => { b.disabled = busy; });
}

// ── 会话历史（localStorage）：只记录拿到 final 的轮次，与服务端历史一致 ──

function loadSessions() {
  try {
    const data = JSON.parse(localStorage.getItem(STORE) ?? '[]');
    return Array.isArray(data) ? data.filter((s) => s && typeof s.id === 'string' && Array.isArray(s.turns)) : [];
  } catch {
    return [];
  }
}

// 超出配额时从最旧的会话开始丢，当前会话保留到最后。
function saveSessions() {
  sessions.sort((a, b) => b.updated - a.updated);
  for (;;) {
    try {
      localStorage.setItem(STORE, JSON.stringify(sessions));
      return;
    } catch {
      const idx = sessions.findLastIndex((s) => s.id !== sessionId);
      if (idx < 0) {
        announce('浏览器存储已满，本轮没有写入会话历史');
        return;
      }
      sessions.splice(idx, 1);
    }
  }
}

function recordTurn(rec) {
  let entry = sessions.find((s) => s.id === sessionId);
  if (!entry) {
    entry = { id: sessionId, title: rec.question.slice(0, 40), turns: [], updated: 0 };
    sessions.push(entry);
  }
  entry.turns.push(rec);
  entry.updated = Date.now();
  saveSessions();
  renderSessions();
}

function renderSessions() {
  const list = $('#sessions');
  list.replaceChildren();
  if (!sessions.length) {
    list.append(el('li', 'hint', '还没有会话'));
    return;
  }
  for (const s of sessions) {
    const li = el('li');
    const open = button('open');
    open.append(el('span', null, s.title || '（无标题）'),
      el('small', null, `${new Date(s.updated).toLocaleString()} · ${s.turns.length} 轮`));
    if (s.id === sessionId) open.setAttribute('aria-current', 'true');
    open.addEventListener('click', () => openSession(s.id));
    const del = button('del', '×', `删除会话：${s.title}`);
    del.addEventListener('click', () => deleteSession(s.id));
    li.append(open, del);
    list.append(li);
  }
}

function resetLog(text) {
  log.replaceChildren();
  const empty = el('p', 'empty', text);
  empty.id = 'empty';
  log.append(empty);
}

function stopStreaming() {
  controller?.abort();
  controller = null;
  setBusy(false);
}

function startNew() {
  stopStreaming();
  sessionId = newSessionId();
  resetLog('新会话已开始。');
  renderSessions();
  announce('新会话已开始');
  input.focus();
}

// 恢复只重绘本地记录；同一 session_id 继续提问时服务端 checkpoint 接着用。
function openSession(id) {
  const entry = sessions.find((s) => s.id === id);
  if (!entry) return;
  stopStreaming();
  sessionId = id;
  log.replaceChildren();
  for (const rec of entry.turns) {
    const turn = addTurn(rec.question, rec.label);
    if (rec.meta) showMeta(turn, rec);
    showFinal(turn, rec);
  }
  renderSessions();
  if (narrow.matches) setSide(false);
  announce(`已打开会话：${entry.title}`);
  input.focus();
}

function deleteSession(id) {
  sessions = sessions.filter((s) => s.id !== id);
  saveSessions();
  if (id === sessionId) startNew();
  else renderSessions();
}

// ── 单轮渲染：流式与恢复共用 showMeta / showFinal ──

function addTurn(question, label) {
  $('#empty')?.remove();
  const prefix = `t${++turnSeq}`;
  const user = el('article', 'msg user');
  user.append(el('h2', 'sr-only', '你的问题'), el('p', 'bubble', question));
  const bot = el('article', 'msg bot');
  const heading = el('h2', 'sr-only', '回答');
  const hard = el('p', 'scope', label || null);
  hard.hidden = !label;
  const stage = el('p', 'stage', '正在连接…');
  const scopeRow = el('div', 'scope-row');
  scopeRow.hidden = true;
  const scope = el('p', 'scope');
  scopeRow.append(scope);
  const body = el('div', 'answer streaming');
  const notice = el('p', 'notice');
  notice.hidden = true;
  const attach = el('section', 'attachments');
  attach.hidden = true;
  const attachTitle = el('h3', null, '相关图片');
  attachTitle.id = `${prefix}-attach`;
  attach.setAttribute('aria-labelledby', attachTitle.id);
  const grid = el('div', 'grid');
  attach.append(attachTitle, grid);
  const sources = el('section', 'sources');
  sources.hidden = true;
  const sourcesTitle = el('h3', null, '来源');
  sourcesTitle.id = `${prefix}-sources`;
  sources.setAttribute('aria-labelledby', sourcesTitle.id);
  const list = el('ol');
  sources.append(sourcesTitle, list);
  const foot = el('div', 'foot');
  bot.append(heading, hard, stage, scopeRow, body, notice, attach, sources, foot);
  log.append(user, bot);
  scrollDown(true);
  return { prefix, stage, scopeRow, scope, body, notice, attach, grid, sources, list, foot };
}

// meta：软过滤说明 +“取消限定”（只在本次检索确实带软过滤时出现）+ 附件图。
function showMeta(turn, rec) {
  const meta = rec.meta;
  if (meta.scope_note) {
    turn.scope.textContent = meta.scope_note;
    turn.scopeRow.hidden = false;
    if (meta.scoped) {
      const again = button('unscope', '取消限定，在全部文档中重新回答');
      again.disabled = controller != null;
      again.addEventListener('click', () => {
        const label = [rec.label, '已取消问题里识别出的范围'].filter(Boolean).join('；');
        if (!controller) ask(rec.question, { filters: rec.filters, label, useScope: false });
      });
      turn.scopeRow.append(again);
    }
  }
  const images = renderAttachments(meta.attachments);
  if (images) {
    turn.grid.innerHTML = images;
    turn.attach.hidden = false;
  }
}

function showFinal(turn, rec) {
  const { final } = rec;
  turn.body.classList.remove('streaming');
  turn.stage.hidden = true;
  turn.body.innerHTML = renderAnswer(rec.answer, final, turn.prefix);
  if (final.notice) {
    turn.notice.textContent = final.notice;
    turn.notice.className = 'notice';
    turn.notice.hidden = false;
  }
  if (final.sources?.length) {
    turn.list.innerHTML = renderSources(final.sources, turn.prefix);
    turn.sources.hidden = false;
  }
  const secs = rec.elapsed != null ? `用时 ${rec.elapsed.toFixed(1)} 秒` : '';
  turn.foot.append(el('span', 'elapsed', secs), feedbackWidget(turn, rec, sessionId));
}

function showFailure(turn, text) {
  turn.body.classList.remove('streaming');
  turn.stage.hidden = true;
  turn.notice.textContent = text;
  turn.notice.className = 'notice error';
  turn.notice.hidden = false;
}

function feedbackWidget(turn, rec, sid) {
  const box = el('div', 'feedback');
  const label = el('span', 'fb-label', '这个回答有帮助吗？');
  const up = button('fb', '👍', '有帮助');
  const down = button('fb', '👎', '没帮助');
  const reasonForm = el('form', 'fb-reason');
  reasonForm.hidden = true;
  const reasonId = `${turn.prefix}-reason`;
  const reasonLabel = el('label', 'sr-only', '没帮助的原因（可选）');
  reasonLabel.htmlFor = reasonId;
  const reason = el('input');
  reason.id = reasonId;
  reason.placeholder = '哪里不对？（可选）';
  const submit = el('button', null, '提交');
  submit.type = 'submit';
  reasonForm.append(reasonLabel, reason, submit);
  const result = el('span', 'fb-result');
  result.setAttribute('role', 'status');

  const done = (rating) => {
    up.disabled = down.disabled = submit.disabled = true;
    (rating > 0 ? up : down).setAttribute('aria-pressed', 'true');
    reasonForm.hidden = true;
  };
  if (rec.rating) {
    done(rec.rating);
    result.textContent = '已反馈';
  }

  async function send(rating, text) {
    up.disabled = down.disabled = submit.disabled = true;
    try {
      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Session-ID': sid },
        body: JSON.stringify({ message_id: rec.final.message_id, rating, ...(text ? { reason: text } : {}) }),
      });
      if (!res.ok) throw new Error(await errorText(res));
      done(rating);
      result.textContent = '已收到反馈，谢谢';
      rec.rating = rating;
      saveSessions();
    } catch (err) {
      up.disabled = down.disabled = submit.disabled = false;
      result.textContent = `反馈失败：${err.message}`;
    }
  }

  up.addEventListener('click', () => send(1));
  down.addEventListener('click', () => {
    reasonForm.hidden = false;
    reason.focus();
  });
  reasonForm.addEventListener('submit', (e) => {
    e.preventDefault();
    send(-1, reason.value.trim());
  });
  box.append(label, up, down, reasonForm, result);
  return box;
}

async function errorText(res) {
  try {
    const data = await res.json();
    if (typeof data.detail === 'string') return data.detail;
    if (Array.isArray(data.detail)) return data.detail.map((d) => d.msg).join('；');
  } catch {
    // 非 JSON 错误体
  }
  return `HTTP ${res.status}`;
}

async function ask(question, { filters = currentFilters(), label = scopeLabel(filters, Object.fromEntries(picked)), useScope = true } = {}) {
  const turn = addTurn(question, label);
  const sid = sessionId;
  const rec = { question, filters, label, use_scope: useScope, answer: '', meta: null, final: null, elapsed: null };
  const started = performance.now();
  let failed = false;
  let pending = false;

  // 流式阶段按帧合并重渲染，长回答不至于每个 token 重排一次。
  const paint = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      if (rec.final) return;
      const stick = nearBottom();
      turn.body.innerHTML = renderAnswer(rec.answer, null, turn.prefix);
      if (stick) scrollDown(true);
    });
  };

  const handle = (ev) => {
    switch (ev.type) {
      case 'progress':
        turn.stage.textContent = `${STAGES[ev.stage] ?? ev.stage}…`;
        announce(turn.stage.textContent);
        break;
      case 'meta':
        rec.meta = { scope_note: ev.scope_note ?? null, scoped: Boolean(ev.scoped), attachments: ev.attachments ?? [] };
        showMeta(turn, rec);
        break;
      case 'token':
        if (!rec.answer) turn.stage.hidden = true;
        rec.answer += ev.text ?? '';
        paint();
        break;
      case 'final':
        rec.final = ev;
        break;
      case 'error':
        failed = true;
        showFailure(turn, ev.message || '服务暂时不可用，请稍后再试');
        break;
      default:
        break;
    }
  };

  const mine = new AbortController();
  controller = mine;
  setBusy(true);
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Session-ID': sid },
      body: JSON.stringify({ question, ...(filters ? { filters } : {}), ...(useScope ? {} : { use_scope: false }) }),
      signal: mine.signal,
    });
    if (!res.ok) throw new Error(await errorText(res));
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
      const parsed = parseSSE(done ? buffer + '\n\n' : buffer);
      buffer = parsed.rest;
      parsed.events.forEach(handle);
      if (done) break;
    }
  } catch (err) {
    failed = true;
    showFailure(turn, err.name === 'AbortError' ? '已停止，本轮不会保存到会话历史。' : `请求失败：${err.message}`);
  } finally {
    // 切换会话会先中止本请求并换上新的 controller，旧请求收尾时不能把它清掉。
    if (controller === mine) {
      controller = null;
      setBusy(false);
    }
  }

  if (rec.final && !mine.signal.aborted) {
    rec.elapsed = (performance.now() - started) / 1000;
    if (sid === sessionId) recordTurn(rec);
    showFinal(turn, rec);
    announce(`回答完成，${rec.final.sources?.length ? `引用 ${rec.final.sources.length} 处资料` : '没有引用资料'}`);
  } else {
    turn.body.classList.remove('streaming');
    turn.stage.hidden = true;
    turn.body.innerHTML = renderAnswer(rec.answer, null, turn.prefix);
    if (!failed) showFailure(turn, '连接提前结束，回答可能不完整。');
    announce('回答未完成');
  }
  scrollDown();
  input.focus();
}

// ── 检索范围（硬过滤）──

function renderTypes() {
  const box = $('#types');
  for (const [name, types] of TYPE_GROUPS) {
    const label = el('label');
    const cb = el('input');
    cb.type = 'checkbox';
    cb.value = types.join(',');
    cb.addEventListener('change', syncScope);
    label.append(cb, document.createTextNode(name));
    box.append(label);
  }
}

function currentFilters() {
  const types = [...document.querySelectorAll('#types input:checked')].flatMap((cb) => cb.value.split(','));
  const filters = {};
  if (types.length) filters.file_types = types;
  if (picked.size) filters.doc_ids = [...picked.keys()];
  return Object.keys(filters).length ? filters : null;
}

function syncScope() {
  const list = $('#picked');
  list.replaceChildren();
  for (const [id, title] of picked) {
    const li = el('li');
    const remove = button(null, '×', `移出范围：${title}`);
    remove.addEventListener('click', () => togglePick(id, title, false));
    li.append(el('span', null, `《${title}》`), remove);
    list.append(li);
  }
  $('#picked-hint').textContent = picked.size ? `文档：已选 ${picked.size} 篇（最多 ${MAX_DOCS}）` : '文档：全部';
  $('#clear-scope').hidden = !currentFilters();
  // 只同步勾选状态，不重建列表，键盘焦点留在原处。
  for (const cb of document.querySelectorAll('#docs input[type="checkbox"]')) {
    cb.checked = picked.has(cb.dataset.id);
    cb.disabled = !cb.checked && picked.size >= MAX_DOCS;
  }
}

function togglePick(id, title, on) {
  if (on && picked.size < MAX_DOCS) picked.set(id, title);
  else if (!on) picked.delete(id);
  syncScope();
}

// ── 文档列表：GET /api/docs，只含已入库文档 ──

async function loadDocs(q = '') {
  const seq = ++docSeq;
  const hint = $('#docs-hint');
  hint.textContent = '加载中…';
  try {
    const res = await fetch('/api/docs' + (q ? `?q=${encodeURIComponent(q)}` : ''));
    if (!res.ok) throw new Error(await errorText(res));
    const data = await res.json();
    if (seq !== docSeq) return;
    renderDocs(data.docs ?? []);
    hint.textContent = !data.docs?.length ? (q ? '没有匹配的文档' : '还没有已入库的文档')
      : data.truncated ? `只显示前 ${data.docs.length} 篇，请按标题搜索` : `共 ${data.docs.length} 篇`;
  } catch (err) {
    if (seq === docSeq) hint.textContent = `文档列表加载失败：${err.message}`;
  }
}

function renderDocs(docs) {
  const list = $('#docs');
  list.replaceChildren();
  for (const d of docs) {
    if (!isDocId(d.doc_id)) continue;
    const title = d.title || d.doc_id;
    const li = el('li');
    const cb = el('input');
    cb.type = 'checkbox';
    cb.id = `doc-${d.doc_id}`;
    cb.dataset.id = d.doc_id;
    cb.addEventListener('change', () => togglePick(d.doc_id, title, cb.checked));
    const label = el('label', null, title);
    label.htmlFor = cb.id;
    label.append(el('small', null, [d.file_type, d.pages ? `${d.pages} 页` : '', `v${d.version}`].filter(Boolean).join(' · ')));
    const view = el('a', 'doc-link', '原文');
    view.href = `/api/docs/${d.doc_id}/file?inline=true`;
    view.target = '_blank';
    view.rel = 'noopener';
    view.setAttribute('aria-label', `在新标签页查看原文：${title}`);
    li.append(cb, label, view);
    list.append(li);
  }
  syncScope();
}

function setSide(open) {
  document.body.classList.toggle('side-closed', !open);
  $('#side').hidden = !open;
  $('#toggle-side').setAttribute('aria-expanded', String(open));
}

// ── 事件绑定与初始化 ──

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question || controller) return;
  input.value = '';
  if (narrow.matches) setSide(false);
  ask(question);
});

// Enter 发送、Shift+Enter 换行；输入法组字时不发送。
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
    e.preventDefault();
    form.requestSubmit();
  }
});

stopBtn.addEventListener('click', () => controller?.abort());
$('#new-chat').addEventListener('click', startNew);
$('#toggle-side').addEventListener('click', () => setSide($('#side').hidden));
$('#clear-scope').addEventListener('click', () => {
  document.querySelectorAll('#types input').forEach((cb) => { cb.checked = false; });
  picked.clear();
  syncScope();
});

let searchTimer = null;
$('#doc-search').addEventListener('input', (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => loadDocs(e.target.value.trim()), 250);
});

renderTypes();
renderSessions();
setSide(!narrow.matches);
loadDocs();

// ?q=... 自动提问，方便截图和冒烟。
const preset = new URLSearchParams(location.search).get('q');
if (preset) ask(preset);
