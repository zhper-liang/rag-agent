// App 级状态不会因空页面转对话页面而卸载；恢复任务、仅在有活动任务时每 2s 轮询。
import { useCallback, useEffect, useRef, useState } from 'react';
import { getJob, listJobs, retryJob, uploadFiles, UploadError } from '../lib/api';
import type { UploadConflict, UploadJob } from '../lib/types';
import { activeJob, cleanFilename, validateFiles } from '../lib/uploads';
import { IconClose, IconRetry } from './icons';

const STATUS: Record<string, string> = { queued: '排队中', cleaning: '清洗中', chunking: '切分中', indexing: '入库中', done: '已完成', failed: '失败', interrupted: '已中断' };
const RESULT: Record<string, string> = { created: '已新增', updated: '已更新', unchanged: '内容未变', duplicate: '内容重复', failed: '失败' };
interface Batch { files: File[]; overwrite: boolean }
interface ConflictState { batch: Batch; conflicts: UploadConflict[]; remaining: Batch[] }
export type ConflictChoice = { action: 'overwrite' | 'rename' | 'skip'; name: string };

export function resolveConflicts(files: File[], conflicts: UploadConflict[], choices: ConflictChoice[]): Batch[] {
  const normal: File[] = [], overwrite: File[] = [];
  const byName = new Map(conflicts.map((c, i) => [c.filename, choices[i]]));
  for (const file of files) {
    const choice = byName.get(cleanFilename(file.name));
    if (!choice) normal.push(file);
    else if (choice.action === 'overwrite') overwrite.push(file);
    else if (choice.action === 'rename') {
      if (!choice.name.trim() || cleanFilename(choice.name) === cleanFilename(file.name)) throw new Error('请为改名文件填写不同的新名称');
      normal.push(new File([file], choice.name.trim(), { type: file.type, lastModified: file.lastModified }));
    }
  }
  const error = normal.length ? validateFiles(normal) : null;
  if (error) throw new Error(error);
  // 混合决策不能给整批带 overwrite=true，覆盖只作用于明确选择的文件。
  const names = [...normal, ...overwrite].map((f) => cleanFilename(f.name));
  if (new Set(names).size !== names.length) throw new Error('改名后文件名重复，请换一个名称');
  return [normal.length ? { files: normal, overwrite: false } : null, overwrite.length ? { files: overwrite, overwrite: true } : null]
    .filter((batch): batch is Batch => batch !== null);
}

export function useUploads(onDone: () => void, notify: (message: string) => void) {
  const [jobs, setJobs] = useState<UploadJob[]>([]);
  const [error, setError] = useState('');
  const [stateError, setStateError] = useState('');
  const [sending, setSending] = useState(false);
  const [conflict, setConflict] = useState<ConflictState | null>(null);
  const [retryBatches, setRetryBatches] = useState<Batch[]>([]);
  const [retrying, setRetrying] = useState<string[]>([]);
  const callbacks = useRef({ onDone, notify });
  callbacks.current = { onDone, notify };
  const lock = useRef(false);
  const mounted = useRef(true);
  const known = useRef(new Map<string, string>());
  const merge = useCallback((incoming: UploadJob[], initial = false) => {
    if (!mounted.current) return;
    for (const job of incoming) {
      const previous = known.current.get(job.job_id);
      known.current.set(job.job_id, job.status);
      if (!initial && job.status === 'done' && previous !== 'done') {
        callbacks.current.onDone();
        callbacks.current.notify('上传任务已完成，文档列表已刷新');
      }
    }
    setJobs((old) => [...incoming, ...old.filter((job) => !incoming.some((j) => j.job_id === job.job_id))]);
  }, []);
  const refresh = useCallback(async (signal?: AbortSignal) => {
    try { merge(await listJobs(signal), true); if (mounted.current) setStateError(''); }
    catch (err) { if (mounted.current && !signal?.aborted) setStateError(`任务列表加载失败：${(err as Error).message}`); }
  }, [merge]);
  useEffect(() => {
    mounted.current = true;
    const ctl = new AbortController();
    void refresh(ctl.signal);
    return () => { mounted.current = false; ctl.abort(); };
  }, [refresh]);
  const activeIds = jobs.filter((job) => activeJob(job.status)).map((job) => job.job_id).sort().join(',');
  useEffect(() => {
    if (!activeIds) return;
    const ctl = new AbortController();
    let timer = 0;
    const poll = async () => {
      const results = await Promise.allSettled(activeIds.split(',').map((id) => getJob(id, ctl.signal)));
      if (ctl.signal.aborted) return;
      const updates = results.flatMap((r) => r.status === 'fulfilled' ? [r.value] : []);
      merge(updates);
      setStateError(results.some((r) => r.status === 'rejected') ? '任务状态暂时无法更新，稍后自动重试' : '');
      timer = window.setTimeout(poll, 2000);
    };
    timer = window.setTimeout(poll, 2000);
    return () => { ctl.abort(); clearTimeout(timer); };
  }, [activeIds, merge]);

  const run = async (batches: Batch[]) => {
    if (lock.current) return;
    lock.current = true; setSending(true); setError(''); setRetryBatches([]);
    try {
      for (let i = 0; i < batches.length; i++) {
        try { merge([await uploadFiles(batches[i].files, batches[i].overwrite)]); }
        catch (err) {
          if (err instanceof UploadError && err.status === 409 && err.conflicts.length) {
            setConflict({ batch: batches[i], conflicts: err.conflicts, remaining: batches.slice(i + 1) });
          } else { setError((err as Error).message); setRetryBatches(batches.slice(i)); }
          return;
        }
      }
    } finally { lock.current = false; if (mounted.current) setSending(false); }
  };
  const submit = (files: File[]) => {
    const problem = validateFiles(files);
    if (problem) { setError(problem); return; }
    void run([{ files, overwrite: false }]);
  };
  const resolve = (choices: ConflictChoice[]) => {
    if (!conflict) return;
    const batches = resolveConflicts(conflict.batch.files, conflict.conflicts, choices);
    setConflict(null);
    void run([...batches, ...conflict.remaining]);
  };
  const retry = async (id: string) => {
    if (retrying.includes(id)) return;
    setRetrying((ids) => [...ids, id]); setError('');
    try { merge([await retryJob(id)]); }
    catch (err) { setError(`重试失败：${(err as Error).message}`); }
    finally { setRetrying((ids) => ids.filter((x) => x !== id)); }
  };
  return { jobs, error: error || stateError, sending, conflict, retryBatches, retrying, submit, resolve, retry,
    cancel: () => { setConflict(null); setRetryBatches([]); setError(''); }, refresh: () => void refresh(), retrySubmit: () => void run(retryBatches) };
}
export type UploadController = ReturnType<typeof useUploads>;

export function UploadPanel({ uploads }: { uploads: UploadController }) {
  const { jobs, error } = uploads;
  const active = jobs.filter((job) => activeJob(job.status)).length;
  if (!jobs.length && !error && !uploads.sending) return null;
  return <section className="upload-panel" aria-label="上传任务">
    {uploads.sending && <p role="status">正在提交上传…</p>}
    {error && <p className="upload-error" role="alert">{error} <button type="button" className="link" onClick={uploads.retryBatches.length ? uploads.retrySubmit : uploads.refresh}>{uploads.retryBatches.length ? '重试提交' : '刷新任务'}</button>{uploads.retryBatches.length > 0 && <> · <button type="button" className="link" onClick={uploads.cancel}>取消这次上传</button></>}</p>}
    {jobs.length > 0 && <details open={active > 0 || jobs.some((j) => ['failed', 'interrupted'].includes(j.status))}>
      <summary>上传任务 · {active ? `${active} 个进行中` : `${jobs.length} 个最近任务`}</summary>
      <ul className="upload-jobs">{jobs.map((job) => <li key={job.job_id}>
        <div className="upload-job-head"><strong>{job.files?.map((f) => f.filename).join('、') || job.job_id}</strong><span role="status">{STATUS[job.status] ?? job.status}</span>
          {['failed', 'interrupted'].includes(job.status) && <button type="button" className="link" disabled={uploads.retrying.includes(job.job_id)} onClick={() => void uploads.retry(job.job_id)}><IconRetry />重试</button>}</div>
        {job.error && <p className="upload-error">{job.error}</p>}
        <ul>{job.files?.map((file, i) => <li key={`${file.filename}-${i}`}>{file.filename}：{RESULT[file.status] ?? file.status}{file.reason ? ` · ${file.reason}` : ''}</li>)}</ul>
      </li>)}</ul>
    </details>}
  </section>;
}

export function ConflictDialog({ state, onResolve, onCancel }: {
  state: ConflictState; onResolve: (choices: ConflictChoice[]) => void; onCancel: () => void;
}) {
  const [choices, setChoices] = useState<ConflictChoice[]>(() => state.conflicts.map((c) => ({ action: 'skip', name: c.filename })));
  const [error, setError] = useState('');
  const dialog = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    dialog.current?.focus();
    return () => previous?.focus();
  }, []);
  const patch = (index: number, change: Partial<ConflictChoice>) => setChoices((old) => old.map((c, i) => i === index ? { ...c, ...change } : c));
  return <div className="modal-backdrop"><div className="conflict-dialog" role="dialog" aria-modal="true" aria-labelledby="conflict-title" tabIndex={-1} ref={dialog}
    onKeyDown={(e) => {
      if (e.key === 'Escape') { e.preventDefault(); onCancel(); }
      if (e.key === 'Tab') {
        const nodes = dialog.current?.querySelectorAll<HTMLElement>('button, select, input');
        if (!nodes?.length) return;
        const first = nodes[0], last = nodes[nodes.length - 1];
        if (e.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && (document.activeElement === last || document.activeElement === dialog.current)) { e.preventDefault(); first.focus(); }
      }
    }}>
    <header><h2 id="conflict-title">处理同名文件</h2><button type="button" className="icon-btn" aria-label="取消上传" onClick={onCancel}><IconClose /></button></header>
    <p>这些文件与已入库文档同名但内容不同。本次尚未写入，请逐个选择；默认跳过，不自动覆盖。</p>
    <form onSubmit={(e) => { e.preventDefault(); try { onResolve(choices); } catch (err) { setError((err as Error).message); } }}>
      {state.conflicts.map((c, i) => <fieldset key={c.filename}><legend>{c.filename}</legend>
        <p>{c.title || c.filename} · 现有版本 v{c.version}</p>
        <label>处理方式<select aria-label={`${c.filename} 的处理方式`} value={choices[i].action} onChange={(e) => patch(i, { action: e.target.value as ConflictChoice['action'] })}>
          <option value="skip">跳过</option><option value="overwrite">覆盖为新版本</option><option value="rename">改名后上传</option>
        </select></label>
        {choices[i].action === 'rename' && <label>新文件名<input required aria-label={`${c.filename} 的新文件名`} value={choices[i].name} onChange={(e) => patch(i, { name: e.target.value })} /></label>}
      </fieldset>)}
      {error && <p className="upload-error" role="alert">{error}</p>}
      <footer><button type="button" className="pill" onClick={onCancel}>取消</button><button type="submit" className="pill active">确认并继续</button></footer>
    </form>
  </div></div>;
}
