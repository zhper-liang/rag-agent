// 单轮问答：问题气泡 + 回答（流式即渲染 Markdown，final 后换引用）+ 来源条 + 操作栏。
import { memo, useMemo, useState } from 'react';
import { sendFeedback } from '../lib/api';
import { renderAnswer, safeImages, sourceMeta, STAGES } from '../lib/render';
import type { Source, TurnRecord } from '../lib/types';
import { IconCopy, IconDoc, IconDownload, IconExternal, IconRetry, IconThumbDown, IconThumbUp } from './icons';

export interface LiveTurn {
  key: string;
  rec: TurnRecord;
  status: 'streaming' | 'done' | 'failed';
  stage: string | null;
  error: string | null;
}

interface Props {
  turn: LiveTurn;
  sessionId: string;
  busy: boolean;
  onRegenerate: (rec: TurnRecord) => void;
  onUnscope: (rec: TurnRecord) => void;
  onOpenSource: (source: Source) => void;
  onRated: (rec: TurnRecord, rating: 1 | -1) => void;
  onCopy: (text: string) => void;
}

export const Turn = memo(function Turn({ turn, sessionId, busy, onRegenerate, onUnscope, onOpenSource, onRated, onCopy }: Props) {
  const { rec, status, stage, error, key } = turn;
  const final = status === 'done' ? rec.final : null;
  const html = useMemo(() => renderAnswer(rec.answer, final, key), [rec.answer, final, key]);
  const sources = final?.sources ?? [];
  const attachments = safeImages(rec.meta?.attachments);

  return (
    <div className="turn">
      <article className="msg user" aria-label="你的问题">
        {rec.label && <p className="scope-label">{rec.label}</p>}
        <p className="bubble">{rec.question}</p>
      </article>
      <article className="msg bot" aria-label="回答" aria-busy={status === 'streaming'}>
        {status === 'streaming' && !rec.answer && (
          <p className="stage"><span className="dots" aria-hidden="true" />{stage ? `${STAGES[stage] ?? stage}…` : '正在连接…'}</p>
        )}
        {rec.meta?.scope_note && (
          <div className="scope-row">
            <span>{rec.meta.scope_note}</span>
            {rec.meta.scoped && status === 'done' && (
              <button type="button" className="link" disabled={busy} onClick={() => onUnscope(rec)}>取消限定，在全部文档中重新回答</button>
            )}
          </div>
        )}
        {rec.answer && (
          <div className={`answer${status === 'streaming' ? ' streaming' : ''}`} dangerouslySetInnerHTML={{ __html: html }} />
        )}
        {final?.notice && <p className="notice">{final.notice}</p>}
        {error && <p className="notice error" role="alert">{error}</p>}
        {attachments.length > 0 && (
          <section className="attachments" aria-label="相关图片">
            {attachments.map((img) => (
              <figure key={img.url}>
                <a href={img.url} target="_blank" rel="noopener"><img src={img.url} alt={img.caption || '相关图片'} loading="lazy" /></a>
                {img.caption && <figcaption>{img.caption}</figcaption>}
              </figure>
            ))}
          </section>
        )}
        {sources.length > 0 && <Sources sources={sources} prefix={key} onOpen={onOpenSource} />}
        {status === 'done' && rec.final && (
          <Actions rec={rec} sessionId={sessionId} busy={busy} onRegenerate={onRegenerate} onRated={onRated} onCopy={onCopy} />
        )}
      </article>
    </div>
  );
});

function Sources({ sources, prefix, onOpen }: { sources: Source[]; prefix: string; onOpen: (s: Source) => void }) {
  const [open, setOpen] = useState(false);
  const titles = [...new Set(sources.map((s) => s.title).filter(Boolean))];
  return (
    <section className={`sources${open ? ' open' : ''}`} aria-label="来源">
      <button type="button" className="sources-bar" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span className="sources-count">{sources.length} 个来源</span>
        <span className="sources-titles">{titles.slice(0, 3).join(' · ')}{titles.length > 3 ? ' …' : ''}</span>
        <span className="chev" aria-hidden="true">{open ? '▴' : '▾'}</span>
      </button>
      {/* 列表始终在 DOM 里，角标 #prefix-src-n 才有落点；收起时只是不显示。 */}
      <ol hidden={!open}>
        {sources.map((s) => {
          const images = safeImages(s.images);
          return (
            <li key={s.cite} id={`${prefix}-src-${s.cite}`}>
              <div className="src-line">
                <span className="src-n">{s.cite}</span>
                <span className="src-title">{s.title}</span>
              </div>
              {sourceMeta(s) && <div className="src-meta">{sourceMeta(s)}</div>}
              {s.doc_id && (
                <div className="src-actions">
                  <button type="button" className="link" onClick={() => onOpen(s)}><IconDoc /> 查看原文</button>
                  <a className="link" href={`/api/docs/${s.doc_id}/file?inline=true`} target="_blank" rel="noopener"><IconExternal /> 新窗口</a>
                  <a className="link" href={`/api/docs/${s.doc_id}/file`} download><IconDownload /> 下载</a>
                </div>
              )}
              {images.length > 0 && (
                <div className="src-images">
                  {images.map((img) => (
                    <a key={img.url} href={img.url} target="_blank" rel="noopener">
                      <img src={img.url} alt={img.caption || `来源 ${s.cite} 的图片`} loading="lazy" />
                    </a>
                  ))}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function Actions({ rec, sessionId, busy, onRegenerate, onRated, onCopy }: {
  rec: TurnRecord; sessionId: string; busy: boolean;
  onRegenerate: (rec: TurnRecord) => void; onRated: (rec: TurnRecord, rating: 1 | -1) => void; onCopy: (text: string) => void;
}) {
  const [reasonOpen, setReasonOpen] = useState(false);
  const [reason, setReason] = useState('');
  const [state, setState] = useState<'idle' | 'sending' | 'error'>('idle');
  const [msg, setMsg] = useState('');
  const messageId = rec.final?.message_id;

  const send = async (rating: 1 | -1, text?: string) => {
    if (!messageId) return;
    setState('sending');
    try {
      await sendFeedback(sessionId, messageId, rating, text);
      setState('idle');
      setReasonOpen(false);
      setMsg('已收到反馈，谢谢');
      onRated(rec, rating);
    } catch (err) {
      setState('error');
      setMsg(`反馈失败：${(err as Error).message}`);
    }
  };

  const rated = rec.rating;
  return (
    <div className="actions">
      <button type="button" className="icon-btn" aria-label="复制回答" title="复制" onClick={() => onCopy(rec.answer)}><IconCopy /></button>
      <button type="button" className="icon-btn" aria-label="重新生成" title="重新生成" disabled={busy} onClick={() => onRegenerate(rec)}><IconRetry /></button>
      {messageId && (
        <>
          <button type="button" className="icon-btn" aria-label="有帮助" title="有帮助" aria-pressed={rated === 1}
            disabled={Boolean(rated) || state === 'sending'} onClick={() => send(1)}><IconThumbUp /></button>
          <button type="button" className="icon-btn" aria-label="没帮助" title="没帮助" aria-pressed={rated === -1}
            disabled={Boolean(rated) || state === 'sending'} onClick={() => setReasonOpen(true)}><IconThumbDown /></button>
        </>
      )}
      {rec.elapsed != null && <span className="elapsed">用时 {rec.elapsed.toFixed(1)} 秒</span>}
      <span className="fb-result" role="status">{msg || (rated ? '已反馈' : '')}</span>
      {reasonOpen && !rated && (
        <form className="fb-reason" onSubmit={(e) => { e.preventDefault(); send(-1, reason.trim()); }}>
          <label className="sr-only" htmlFor={`reason-${messageId}`}>没帮助的原因（可选）</label>
          <input id={`reason-${messageId}`} placeholder="哪里不对？（可选）" maxLength={500} autoFocus
            value={reason} onChange={(e) => setReason(e.target.value)} />
          <button type="submit" disabled={state === 'sending'}>提交</button>
          <button type="button" className="link" onClick={() => setReasonOpen(false)}>取消</button>
        </form>
      )}
    </div>
  );
}
