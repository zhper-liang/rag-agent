// 原文抽屉：页内渲染 full.md（同一渲染器，HTML 只转义），从引用打开时定位到 section。
import { useEffect, useMemo, useRef, useState } from 'react';
import { fetchDocText } from '../lib/api';
import { findSection, renderDoc } from '../lib/render';
import { IconClose, IconDownload, IconExternal } from './icons';

interface Props {
  docId: string;
  section: string;
  title: string;
  onClose: () => void;
}

export function DocDrawer({ docId, section, title, onClose }: Props) {
  const [doc, setDoc] = useState<{ text: string; version: number } | null>(null);
  const [error, setError] = useState('');
  const body = useRef<HTMLDivElement>(null);
  const closeBtn = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const ctl = new AbortController();
    setDoc(null);
    setError('');
    fetchDocText(docId, ctl.signal).then(setDoc).catch((err) => {
      if (err.name !== 'AbortError') setError(err.message);
    });
    return () => ctl.abort();
  }, [docId]);

  useEffect(() => { closeBtn.current?.focus(); }, [docId]);

  useEffect(() => {
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', esc);
    return () => document.removeEventListener('keydown', esc);
  }, [onClose]);

  const html = useMemo(() => (doc ? renderDoc(doc.text, docId, doc.version) : ''), [doc, docId]);
  const heading = title || doc?.text.match(/^#\s+(.+)$/m)?.[1] || docId;

  // 渲染完再定位：整路径优先，退到最后一级标题；找不到就停在开头。
  useEffect(() => {
    const root = body.current;
    if (!root || !html) return;
    root.querySelectorAll('.located').forEach((el) => el.classList.remove('located'));
    if (!section) { root.scrollTop = 0; return; }
    const heads = [...root.querySelectorAll<HTMLElement>('[data-path]')];
    const idx = findSection(heads.map((h) => h.dataset.path ?? ''), section);
    const target = heads[idx];
    if (target) {
      target.classList.add('located');
      target.scrollIntoView({ block: 'start' });
    }
  }, [html, section]);

  return (
    <aside className="drawer" aria-label={`原文：${heading}`}>
      <header className="drawer-head">
        <div className="drawer-title">
          <h2>{heading}</h2>
          {section && <p className="drawer-section">{section}</p>}
        </div>
        <a className="icon-btn" href={`/api/docs/${docId}/file?inline=true`} target="_blank" rel="noopener"
          aria-label="新窗口查看纯文本" title="新窗口"><IconExternal /></a>
        <a className="icon-btn" href={`/api/docs/${docId}/file`} download aria-label="下载原文" title="下载"><IconDownload /></a>
        <button type="button" ref={closeBtn} className="icon-btn" aria-label="关闭原文" onClick={onClose}><IconClose /></button>
      </header>
      <div className="drawer-body" ref={body}>
        {error && <p className="notice error" role="alert">{error}</p>}
        {!doc && !error && <p className="hint">加载中…</p>}
        {doc && <div className="answer doc" dangerouslySetInnerHTML={{ __html: html }} />}
      </div>
    </aside>
  );
}
