// 页面骨架：侧栏 + 对话 + 原文抽屉。路由 / /c/{id} /d/{doc}；会话只存 localStorage。
import { useCallback, useEffect, useRef, useState } from 'react';
import { checkReady } from './lib/api';
import { GREY, scopeLabel } from './lib/render';
import { canGoBack, navigate, useRoute } from './lib/router';
import { appendTurn, loadSessions, newSessionId, patchSession, saveSessions } from './lib/sessions';
import { streamChat } from './lib/sse';
import type { Doc, Session, Source, TurnRecord } from './lib/types';
import { Composer, scopeFilters, type Scope } from './components/Composer';
import { DocDrawer } from './components/DocDrawer';
import { IconLogo, IconPlus, IconSidebar } from './components/icons';
import { Sidebar } from './components/Sidebar';
import { Turn, type LiveTurn } from './components/Turn';

const narrowQuery = window.matchMedia('(max-width: 860px)');
let turnSeq = 0;
const nextKey = () => `t${++turnSeq}`;
const fromRecord = (rec: TurnRecord): LiveTurn => ({ key: nextKey(), rec, status: 'done', stage: null, error: null });

interface Tip { x: number; y: number; title: string; meta: string; hint: string }

export function App() {
  const route = useRoute();
  const [sessions, setSessions] = useState<Session[]>(loadSessions);
  const [sessionId, setSessionId] = useState(() => (route.name === 'chat' ? route.sessionId : newSessionId()));
  const [turns, setTurns] = useState<LiveTurn[]>(() => {
    const s = route.name === 'chat' ? loadSessions().find((x) => x.id === route.sessionId) : undefined;
    return (s?.turns ?? []).map(fromRecord);
  });
  const [busy, setBusy] = useState(false);
  const [scope, setScope] = useState<Scope>({ types: [], docs: new Map() });
  const [sideOpen, setSideOpen] = useState(!narrowQuery.matches);
  const [ready, setReady] = useState<'ok' | 'not_ready' | 'unknown'>('unknown');
  const [tip, setTip] = useState<Tip | null>(null);
  const [toast, setToast] = useState('');
  const [announce, setAnnounce] = useState('');
  const [docTitles, setDocTitles] = useState<Record<string, string>>({});
  const controller = useRef<AbortController | null>(null);
  const log = useRef<HTMLDivElement>(null);
  const sessionRef = useRef(sessionId);
  sessionRef.current = sessionId;

  const persist = useCallback((next: Session[]) => {
    const { saved, full } = saveSessions(next, sessionRef.current);
    setSessions(saved);
    if (full) setAnnounce('浏览器存储已满，本轮没有写入会话历史');
  }, []);

  const stop = useCallback(() => {
    controller.current?.abort();
    controller.current = null;
    setBusy(false);
  }, []);

  // 路由 → 会话：/c/{id} 打开（本地没有也沿用该 id，服务端 checkpoint 接着用）；/ 开新会话；/d/ 不动对话。
  useEffect(() => {
    if (route.name === 'doc') return;
    const want = route.name === 'chat' ? route.sessionId : null;
    if (want === sessionRef.current) return;
    if (route.name === 'home' && !turns.length) return;
    stop();
    const id = want ?? newSessionId();
    setSessionId(id);
    setTurns((sessions.find((s) => s.id === id)?.turns ?? []).map(fromRecord));
    if (narrowQuery.matches) setSideOpen(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [route]);

  // 就绪检查：编码服务加载模型期间提示，好了就停。旧进程没有该接口时按未知处理。
  useEffect(() => {
    let timer = 0;
    const poll = async () => {
      const state = await checkReady();
      setReady(state);
      if (state === 'not_ready') timer = window.setTimeout(poll, 10_000);
    };
    poll();
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const on = () => setSideOpen(!narrowQuery.matches);
    narrowQuery.addEventListener('change', on);
    return () => narrowQuery.removeEventListener('change', on);
  }, []);

  const flash = (text: string) => {
    setToast(text);
    window.setTimeout(() => setToast(''), 1600);
  };

  const copy = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      flash('已复制');
    } catch {
      flash('复制失败，浏览器不允许访问剪贴板');
    }
  }, []);

  const nearBottom = () => {
    const el = log.current;
    return !el || el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  };
  const scrollDown = () => requestAnimationFrame(() => { if (log.current) log.current.scrollTop = log.current.scrollHeight; });
  const sessionsRef = useRef(sessions);
  sessionsRef.current = sessions;
  const routeRef = useRef(route);
  routeRef.current = route;
  const busyRef = useRef(busy);
  busyRef.current = busy;

  const updateTurn = (key: string, fn: (t: LiveTurn) => LiveTurn) =>
    setTurns((ts) => ts.map((t) => (t.key === key ? fn(t) : t)));

  const ask = useCallback(async (question: string, opts: { filters: TurnRecord['filters']; label: string; useScope: boolean }) => {
    if (busyRef.current) return;
    const sid = sessionRef.current;
    const key = nextKey();
    const rec: TurnRecord = { question, filters: opts.filters, label: opts.label, use_scope: opts.useScope, answer: '', meta: null, final: null, elapsed: null };
    setTurns((ts) => [...ts, { key, rec: { ...rec }, status: 'streaming', stage: null, error: null }]);
    if (routeRef.current.name === 'home') navigate({ name: 'chat', sessionId: sid });
    scrollDown();

    // token 按帧合并：长回答不至于每个 token 重排一次。
    let pending = false;
    let stage: string | null = null;
    let error: string | null = null;
    const flush = () => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        const stick = nearBottom();
        updateTurn(key, (t) => (t.status === 'streaming' ? { ...t, rec: { ...rec }, stage, error } : t));
        if (stick) scrollDown();
      });
    };

    const mine = new AbortController();
    controller.current = mine;
    setBusy(true);
    const started = performance.now();
    try {
      await streamChat({ question, sessionId: sid, filters: opts.filters, useScope: opts.useScope }, (ev) => {
        switch (ev.type) {
          case 'progress': stage = String(ev.stage ?? ''); break;
          case 'meta':
            rec.meta = { scope_note: (ev.scope_note as string) ?? null, scoped: Boolean(ev.scoped), attachments: (ev.attachments as []) ?? [] };
            break;
          case 'token': rec.answer += String(ev.text ?? ''); break;
          case 'final': rec.final = ev as TurnRecord['final']; break;
          case 'error': error = String(ev.message || '服务暂时不可用，请稍后再试'); break;
          default: return;
        }
        flush();
      }, mine.signal);
    } catch (err) {
      error = (err as Error).name === 'AbortError' ? '已停止，本轮不会保存到会话历史。' : `请求失败：${(err as Error).message}`;
    } finally {
      // 切换会话会先中止本请求并换上新的 controller，旧请求收尾时不能把它清掉。
      if (controller.current === mine) {
        controller.current = null;
        setBusy(false);
      }
    }

    const ok = Boolean(rec.final) && !mine.signal.aborted && !error;
    if (ok) {
      rec.elapsed = (performance.now() - started) / 1000;
      if (sid === sessionRef.current) persist(appendTurn(sessionsRef.current, sid, rec));
      const n = rec.final?.sources?.length ?? 0;
      setAnnounce(`回答完成，${n ? `引用 ${n} 处资料` : '没有引用资料'}`);
    } else {
      if (!error) error = rec.final ? '回答未完成。' : '连接提前结束，回答可能不完整。';
      setAnnounce('回答未完成');
    }
    updateTurn(key, (t) => ({ ...t, rec: { ...rec }, status: ok ? 'done' : 'failed', stage: null, error: ok ? null : error }));
  }, [persist]);

  const send = useCallback((question: string) => {
    const filters = scopeFilters(scope);
    ask(question, { filters, label: scopeLabel(filters, Object.fromEntries(scope.docs)), useScope: true });
  }, [ask, scope]);

  const regenerate = useCallback((rec: TurnRecord) => {
    ask(rec.question, { filters: rec.filters, label: rec.label, useScope: rec.use_scope });
  }, [ask]);

  const unscope = useCallback((rec: TurnRecord) => {
    const label = [rec.label, '已取消问题里识别出的范围'].filter(Boolean).join('；');
    ask(rec.question, { filters: rec.filters, label, useScope: false });
  }, [ask]);

  const rated = useCallback((rec: TurnRecord, rating: 1 | -1) => {
    const mid = rec.final?.message_id;
    setTurns((ts) => ts.map((t) => (t.rec.final?.message_id === mid ? { ...t, rec: { ...t.rec, rating } } : t)));
    const sid = sessionRef.current;
    persist(sessionsRef.current.map((s) => (s.id !== sid ? s : {
      ...s, turns: s.turns.map((r) => (r.final?.message_id === mid ? { ...r, rating } : r)),
    })));
  }, [persist]);

  const openDoc = useCallback((docId: string, section: string, title: string) => {
    if (title) setDocTitles((m) => ({ ...m, [docId]: title }));
    navigate({ name: 'doc', docId, section }, { back: true });
  }, []);
  const openSource = useCallback((s: Source) => { if (s.doc_id) openDoc(s.doc_id, s.section ?? '', s.title ?? ''); }, [openDoc]);
  const openPicked = useCallback((d: Doc) => openDoc(d.doc_id, '', d.title), [openDoc]);

  // 关原文：有对话就回到 /c/{id}（replace，避免回到 / 时被当成新会话），否则回首页。
  const closeDoc = useCallback(() => {
    if (turns.length) navigate({ name: 'chat', sessionId: sessionRef.current }, { replace: true });
    else if (canGoBack()) history.back();
    else navigate({ name: 'home' }, { replace: true });
  }, [turns.length]);

  const startNew = useCallback(() => {
    stop();
    const id = newSessionId();
    setSessionId(id);
    sessionRef.current = id;
    setTurns([]);
    navigate({ name: 'home' });
    if (narrowQuery.matches) setSideOpen(false);
  }, [stop]);

  const openSession = useCallback((id: string) => navigate({ name: 'chat', sessionId: id }), []);
  const rename = useCallback((id: string, title: string) => persist(patchSession(sessionsRef.current, id, { title, renamed: true })), [persist]);
  const pin = useCallback((id: string, pinned: boolean) => persist(patchSession(sessionsRef.current, id, { pinned })), [persist]);
  const remove = useCallback((id: string) => {
    persist(sessionsRef.current.filter((s) => s.id !== id));
    if (id === sessionRef.current) startNew();
  }, [persist, startNew]);

  // 回答区事件委托：角标悬停预览、点击打开原文；代码块复制。
  const sourceOf = (el: HTMLElement): Source | undefined => {
    const t = turns.find((x) => x.key === el.dataset.turn);
    return t?.rec.final?.sources?.find((s) => Number(s.cite) === Number(el.dataset.cite));
  };
  const showTip = (el: HTMLElement) => {
    const s = sourceOf(el);
    const hint = el.dataset.hint ?? '';
    if (!s && !hint) return;
    const r = el.getBoundingClientRect();
    setTip({ x: r.left + r.width / 2, y: r.top, title: s?.title ?? '', meta: s ? [s.section, s.pages ? `第 ${s.pages} 页` : ''].filter(Boolean).join(' · ') : '', hint });
  };
  const onLogOver = (e: React.MouseEvent | React.FocusEvent) => {
    const el = (e.target as HTMLElement).closest<HTMLElement>('.cite');
    if (el) showTip(el);
  };
  const onLogOut = (e: React.MouseEvent | React.FocusEvent) => {
    if ((e.target as HTMLElement).closest('.cite')) setTip(null);
  };
  const onLogClick = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    const copyBtn = target.closest<HTMLElement>('.copy-code');
    if (copyBtn) {
      copy(copyBtn.closest('.code')?.querySelector('code')?.textContent ?? '');
      return;
    }
    const el = target.closest<HTMLElement>('.cite');
    if (!el) return;
    const s = sourceOf(el);
    if (s?.doc_id) {
      e.preventDefault();
      setTip(null);
      openSource(s);
    }
  };

  const current = sessions.find((s) => s.id === sessionId);
  const empty = !turns.length;
  const composer = (
    <Composer busy={busy} scope={scope} onScope={setScope} onSend={send} onStop={stop} onOpenDoc={openPicked}
      placeholder="输入问题，Enter 发送，Shift+Enter 换行" />
  );

  return (
    <div className={`app${sideOpen ? ' side-open' : ''}${route.name === 'doc' ? ' doc-open' : ''}`}>
      {sideOpen && (
        <Sidebar sessions={sessions} currentId={sessionId} onNew={startNew} onOpen={openSession}
          onRename={rename} onPin={pin} onDelete={remove} onCollapse={() => setSideOpen(false)} />
      )}
      {sideOpen && <div className="scrim" aria-hidden="true" onClick={() => setSideOpen(false)} />}
      <main className="main">
        <header className="top">
          {!sideOpen && (
            <>
              <button type="button" className="icon-btn" aria-label="展开侧栏" onClick={() => setSideOpen(true)}><IconSidebar /></button>
              <button type="button" className="icon-btn" aria-label="开启新对话" onClick={startNew}><IconPlus /></button>
            </>
          )}
          <h1 className="top-title">{empty ? '' : (current?.title || turns[0]?.rec.question.slice(0, 40) || '')}</h1>
        </header>
        {ready === 'not_ready' && (
          <p className="banner" role="status">服务正在启动或依赖未就绪（Milvus / 编码服务 / 登记表），提问可能失败，稍后自动恢复。</p>
        )}
        {empty ? (
          <div className="hero">
            <h2 className="hero-title"><IconLogo /> 你好，有什么想从资料里查的？</h2>
            {composer}
            <p className="hero-hint">回答里的角标指向来源；灰色角标表示{GREY.unsupported}。</p>
          </div>
        ) : (
          <>
            <div className="log" ref={log} aria-label="对话记录"
              onMouseOver={onLogOver} onMouseOut={onLogOut} onFocus={onLogOver} onBlur={onLogOut} onClick={onLogClick}>
              <div className="log-inner">
                {turns.map((t) => (
                  <Turn key={t.key} turn={t} sessionId={sessionId} busy={busy} onRegenerate={regenerate}
                    onUnscope={unscope} onOpenSource={openSource} onRated={rated} onCopy={copy} />
                ))}
              </div>
            </div>
            <div className="dock">{composer}<p className="dock-hint">内容由模型根据资料生成，请以原文为准。</p></div>
          </>
        )}
      </main>
      {route.name === 'doc' && (
        <DocDrawer docId={route.docId} section={route.section} title={docTitles[route.docId] ?? ''} onClose={closeDoc} />
      )}
      {tip && (
        <div className="tip" role="tooltip" style={{ left: tip.x, top: tip.y }}>
          {tip.title && <strong>{tip.title}</strong>}
          {tip.meta && <span>{tip.meta}</span>}
          {tip.hint && <em>{tip.hint}</em>}
        </div>
      )}
      {toast && <div className="toast" role="status">{toast}</div>}
      <div className="sr-only" role="status" aria-live="polite">{announce}</div>
    </div>
  );
}
