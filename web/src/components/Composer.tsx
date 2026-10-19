// 输入框 + 检索范围（文件类型、文档，作为硬过滤 filters 发送）。Enter 发送，Shift+Enter 换行，输入法组字时不发送。
import { useEffect, useRef, useState } from 'react';
import { listDocs } from '../lib/api';
import { uploadAccept, uploadConfig } from '../lib/uploads';
import type { Doc, Filters, TimeRange } from '../lib/types';
import { IconArrowUp, IconAttach, IconClose, IconDoc, IconFilter, IconStop } from './icons';

// 与 retrieval/filters.py 的 FILE_TYPES、MAX_FILTER_ITEMS 一致。
export const TYPE_GROUPS: [string, string[]][] = [
  ['PDF', ['pdf']], ['Word', ['doc', 'docx']], ['PPT', ['ppt', 'pptx']],
  ['图片', ['png', 'jpg', 'jpeg']], ['文本', ['md', 'txt']],
];
export const MAX_DOCS = 100;

export interface Scope extends TimeRange { types: string[]; docs: Map<string, string> }

export function scopeFilters(scope: Scope): Filters | null {
  const file_types = TYPE_GROUPS.filter(([label]) => scope.types.includes(label)).flatMap(([, t]) => t);
  const doc_ids = [...scope.docs.keys()];
  if (!file_types.length && !doc_ids.length) return null;
  return { ...(file_types.length ? { file_types } : {}), ...(doc_ids.length ? { doc_ids } : {}) };
}

interface Props {
  busy: boolean;
  scope: Scope;
  onScope: (scope: Scope) => void;
  onSend: (question: string) => void;
  onStop: () => void;
  onOpenDoc: (doc: Doc) => void;
  placeholder: string;
  onUpload: (files: File[]) => void;
  uploading: boolean;
  docsRevision: number;
}

export function Composer({ busy, scope, onScope, onSend, onStop, onOpenDoc, placeholder, onUpload, uploading, docsRevision }: Props) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [text, setText] = useState('');
  const [picking, setPicking] = useState(false);
  const area = useRef<HTMLTextAreaElement>(null);
  const count = scope.types.length + scope.docs.size + Number(Boolean(scope.time_from || scope.time_to));
  const invalidTime = Boolean(scope.time_from && scope.time_to && scope.time_from >= scope.time_to);

  // 高度随内容增长，最多 8 行左右，超出滚动。
  useEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = 'auto';
    // 空内容不量：首帧布局未定时占位文字会按窄宽度折行，量出来偏高。
    if (!text) return;
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [text]);

  const submit = () => {
    const q = text.trim();
    if (!q || busy || invalidTime) return;
    onSend(q);
    setText('');
  };

  return (
    <form className="composer" onSubmit={(e) => { e.preventDefault(); submit(); }}>
      <label htmlFor="question" className="sr-only">问题</label>
      <textarea id="question" ref={area} rows={2} value={text} placeholder={placeholder}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            submit();
          }
        }} />
      <div className="composer-bar">
        <div className="composer-left">
          <input ref={fileInput} type="file" multiple accept={uploadAccept} hidden aria-label="选择上传文件"
            onChange={(e) => { const files = Array.from(e.target.files ?? []); e.target.value = ''; if (files.length) onUpload(files); }} />
          <button type="button" className="icon-btn" aria-label="上传文档" disabled={uploading}
            title={`上传文档（最多 ${uploadConfig.maxFiles} 个，单个 ${uploadConfig.maxFileMb} MB）`} onClick={() => fileInput.current?.click()}><IconAttach /></button>
          <button type="button" className={`pill${count ? ' active' : ''}`} aria-expanded={picking}
            aria-controls="scope-picker" onClick={() => setPicking(!picking)}>
            <IconFilter /> {count ? `检索范围 · ${count}` : '检索范围'}
          </button>
          {[...scope.docs].slice(0, 3).map(([id, title]) => (
            <span key={id} className="chip">
              <span className="chip-text">{title}</span>
              <button type="button" aria-label={`移出范围：${title}`}
                onClick={() => { const docs = new Map(scope.docs); docs.delete(id); onScope({ ...scope, docs }); }}><IconClose /></button>
            </span>
          ))}
          {scope.docs.size > 3 && <span className="chip muted">+{scope.docs.size - 3}</span>}
        </div>
        {busy ? (
          <button type="button" className="send stop" aria-label="停止生成" onClick={onStop}><IconStop /></button>
        ) : (
          <button type="submit" className="send" aria-label="发送" disabled={!text.trim() || invalidTime}><IconArrowUp /></button>
        )}
      </div>
      {invalidTime && <p className="upload-error" role="alert">结束日期必须晚于开始日期（结束日期不含当天）。</p>}
      {picking && <ScopePicker docsRevision={docsRevision} scope={scope} onScope={onScope} onClose={() => setPicking(false)} onOpenDoc={onOpenDoc} />}
    </form>
  );
}

function ScopePicker({ scope, onScope, onClose, onOpenDoc, docsRevision }: {
  docsRevision: number;
  scope: Scope; onScope: (s: Scope) => void; onClose: () => void; onOpenDoc: (doc: Doc) => void;
}) {
  const [q, setQ] = useState('');
  const [docs, setDocs] = useState<Doc[]>([]);
  const [hint, setHint] = useState('加载中…');
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ctl = new AbortController();
    const timer = setTimeout(() => {
      listDocs(q, ctl.signal).then(({ docs, truncated }) => {
        setDocs(docs);
        setHint(docs.length ? `${q ? '匹配' : '共'} ${docs.length}${truncated ? '+' : ''} 篇` : (q ? '没有匹配的文档' : '还没有已入库的文档'));
      }).catch((err) => { if (err.name !== 'AbortError') setHint(`文档列表加载失败：${err.message}`); });
    }, q ? 250 : 0);
    return () => { clearTimeout(timer); ctl.abort(); };
  }, [q, docsRevision]);

  useEffect(() => {
    const close = (e: Event) => {
      const t = e.target as HTMLElement;
      if (!box.current?.contains(t) && !t.closest('.pill')) onClose();
    };
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('pointerdown', close);
    document.addEventListener('keydown', esc);
    return () => { document.removeEventListener('pointerdown', close); document.removeEventListener('keydown', esc); };
  }, [onClose]);

  const toggleType = (label: string, on: boolean) =>
    onScope({ ...scope, types: on ? [...scope.types, label] : scope.types.filter((t) => t !== label) });
  const toggleDoc = (d: Doc, on: boolean) => {
    const next = new Map(scope.docs);
    if (on && next.size < MAX_DOCS) next.set(d.doc_id, d.title || d.doc_id);
    if (!on) next.delete(d.doc_id);
    onScope({ ...scope, docs: next });
  };

  return (
    <div className="scope-picker" id="scope-picker" ref={box} role="dialog" aria-label="检索范围">
      <fieldset className="types">
        <legend>文件类型（不选为全部）</legend>
        {TYPE_GROUPS.map(([label]) => (
          <label key={label} className="check">
            <input type="checkbox" checked={scope.types.includes(label)} onChange={(e) => toggleType(label, e.target.checked)} />
            {label}
          </label>
        ))}
      </fieldset>
      <fieldset className="time-range"><legend>入库时间（含开始，不含结束）</legend>
        <label>开始日期<input type="date" aria-label="入库开始日期" value={scope.time_from ?? ''}
          max={scope.time_to || undefined} onChange={(e) => onScope({ ...scope, time_from: e.target.value || undefined })} /></label>
        <label>结束日期（不含）<input type="date" aria-label="入库结束日期" value={scope.time_to ?? ''}
          min={scope.time_from || undefined} onChange={(e) => onScope({ ...scope, time_to: e.target.value || undefined })} /></label>
      </fieldset>
      <div className="docs-head">
        <span>文档（不选为全部，最多 {MAX_DOCS} 篇）</span>
        {(scope.docs.size > 0 || scope.types.length > 0 || scope.time_from || scope.time_to) && (
          <button type="button" className="link" onClick={() => onScope({ types: [], docs: new Map() })}>清空范围</button>
        )}
      </div>
      <input type="search" className="doc-search" placeholder="按标题搜索" aria-label="按标题搜索文档"
        value={q} onChange={(e) => setQ(e.target.value)} />
      <p className="hint" role="status">{hint}</p>
      <ul className="doc-list">
        {docs.map((d) => (
          <li key={d.doc_id}>
            <label className="check">
              <input type="checkbox" checked={scope.docs.has(d.doc_id)}
                disabled={!scope.docs.has(d.doc_id) && scope.docs.size >= MAX_DOCS}
                onChange={(e) => toggleDoc(d, e.target.checked)} />
              <span className="doc-title">{d.title || d.doc_id}</span>
              <small>{d.file_type} · v{d.version}<br />入库：{d.indexed_at ? d.indexed_at.slice(0, 10) : '未知'}</small>
            </label>
            <button type="button" className="icon-btn" aria-label={`查看原文：${d.title}`} onClick={() => onOpenDoc(d)}><IconDoc /></button>
          </li>
        ))}
      </ul>
    </div>
  );
}
