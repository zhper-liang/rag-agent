// 演示页 DOM 层：POST /api/chat 读 SSE，final 后重渲染引用；反馈走 /api/feedback。
import { STAGES, parseSSE, renderAnswer, renderSources } from './render.js';

const $ = (sel) => document.querySelector(sel);
const log = $('#log');
const form = $('#ask');
const input = $('#question');
const sendBtn = $('#send');
const stopBtn = $('#stop');
const status = $('#status');

const newSessionId = () => {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return 'web-' + Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
};

let sessionId = newSessionId();
let controller = null;
let turnSeq = 0;

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
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
}

function addTurn(question) {
  $('#empty')?.remove();
  const prefix = `t${++turnSeq}`;
  const user = el('article', 'msg user');
  user.append(el('h2', 'sr-only', '你的问题'), el('p', 'bubble', question));
  const bot = el('article', 'msg bot');
  const heading = el('h2', 'sr-only', '回答');
  const stage = el('p', 'stage', '正在连接…');
  const scope = el('p', 'scope');
  scope.hidden = true;
  const body = el('div', 'answer streaming');
  const notice = el('p', 'notice');
  notice.hidden = true;
  const sources = el('section', 'sources');
  sources.hidden = true;
  const sourcesTitle = el('h3', null, '来源');
  sourcesTitle.id = `${prefix}-sources`;
  sources.setAttribute('aria-labelledby', sourcesTitle.id);
  const list = el('ol');
  sources.append(sourcesTitle, list);
  const foot = el('div', 'foot');
  bot.append(heading, stage, scope, body, notice, sources, foot);
  log.append(user, bot);
  scrollDown(true);
  return { prefix, stage, scope, body, notice, sources, list, foot };
}

function feedbackWidget(turn, messageId, sid) {
  const box = el('div', 'feedback');
  const label = el('span', 'fb-label', '这个回答有帮助吗？');
  const up = el('button', 'fb', '👍');
  up.type = 'button';
  up.setAttribute('aria-label', '有帮助');
  const down = el('button', 'fb', '👎');
  down.type = 'button';
  down.setAttribute('aria-label', '没帮助');
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

  async function send(rating, text) {
    up.disabled = down.disabled = submit.disabled = true;
    try {
      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Session-ID': sid },
        body: JSON.stringify({ message_id: messageId, rating, ...(text ? { reason: text } : {}) }),
      });
      if (!res.ok) throw new Error(await errorText(res));
      (rating > 0 ? up : down).setAttribute('aria-pressed', 'true');
      reasonForm.hidden = true;
      result.textContent = '已收到反馈，谢谢';
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

async function ask(question) {
  const turn = addTurn(question);
  const sid = sessionId;
  const started = performance.now();
  let answer = '';
  let final = null;
  let failed = false;
  let pending = false;

  // 流式阶段按帧合并重渲染，长回答不至于每个 token 重排一次。
  const paint = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      if (final) return;
      const stick = nearBottom();
      turn.body.innerHTML = renderAnswer(answer, null, turn.prefix);
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
        if (ev.scope_note) {
          turn.scope.textContent = ev.scope_note;
          turn.scope.hidden = false;
        }
        break;
      case 'token':
        if (!answer) turn.stage.hidden = true;
        answer += ev.text ?? '';
        paint();
        break;
      case 'final':
        final = ev;
        break;
      case 'error':
        failed = true;
        turn.stage.hidden = true;
        turn.notice.textContent = ev.message || '服务暂时不可用，请稍后再试';
        turn.notice.className = 'notice error';
        turn.notice.hidden = false;
        break;
      default:
        break;
    }
  };

  controller = new AbortController();
  setBusy(true);
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Session-ID': sid },
      body: JSON.stringify({ question }),
      signal: controller.signal,
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
    turn.stage.hidden = true;
    turn.notice.textContent = err.name === 'AbortError' ? '已停止，本轮不会保存到会话历史。' : `请求失败：${err.message}`;
    turn.notice.className = 'notice error';
    turn.notice.hidden = false;
  } finally {
    controller = null;
    setBusy(false);
  }

  turn.body.classList.remove('streaming');
  turn.stage.hidden = true;
  if (final) {
    turn.body.innerHTML = renderAnswer(answer, final, turn.prefix);
    if (final.notice) {
      turn.notice.textContent = final.notice;
      turn.notice.className = 'notice';
      turn.notice.hidden = false;
    }
    if (final.sources?.length) {
      turn.list.innerHTML = renderSources(final.sources, turn.prefix);
      turn.sources.hidden = false;
    }
    const secs = ((performance.now() - started) / 1000).toFixed(1);
    turn.foot.append(el('span', 'elapsed', `用时 ${secs} 秒`), feedbackWidget(turn, final.message_id, sid));
    announce(`回答完成，${final.sources?.length ? `引用 ${final.sources.length} 处资料` : '没有引用资料'}`);
  } else {
    turn.body.innerHTML = renderAnswer(answer, null, turn.prefix);
    if (!failed) {
      turn.notice.textContent = '连接提前结束，回答可能不完整。';
      turn.notice.className = 'notice error';
      turn.notice.hidden = false;
    }
    announce('回答未完成');
  }
  scrollDown();
  input.focus();
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question || controller) return;
  input.value = '';
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

$('#new-chat').addEventListener('click', () => {
  controller?.abort();
  sessionId = newSessionId();
  log.replaceChildren();
  const empty = el('p', 'empty', '新会话已开始。');
  empty.id = 'empty';
  log.append(empty);
  announce('新会话已开始');
  input.focus();
});

// ?q=... 自动提问，方便截图和冒烟。
const preset = new URLSearchParams(location.search).get('q');
if (preset) ask(preset);
