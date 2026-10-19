// 其余同源接口：文档列表、原文、反馈、就绪检查。
import { errorText } from './sse';
import { isDocId } from './render';
import { cleanFilename, uploadConfig, validateFiles } from './uploads';
import type { Doc, UploadConflict, UploadFileResult, UploadJob } from './types';

export async function listDocs(q: string, signal?: AbortSignal): Promise<{ docs: Doc[]; truncated: boolean }> {
  const res = await fetch(`/api/docs${q.trim() ? `?q=${encodeURIComponent(q.trim())}` : ''}`, { signal });
  if (!res.ok) throw new Error(await errorText(res));
  const data = await res.json();
  return { docs: Array.isArray(data.docs) ? data.docs.filter((d: Doc) => isDocId(d.doc_id)) : [], truncated: Boolean(data.truncated) };
}

// 版本号来自 X-Doc-Version 响应头：图片地址 /api/images/{doc}/v{n}/ 要用登记表里的有效版本。
export async function fetchDocText(docId: string, signal?: AbortSignal): Promise<{ text: string; version: number }> {
  if (!isDocId(docId)) throw new Error('非法 doc_id');
  const res = await fetch(`/api/docs/${docId}/file`, { signal });
  if (!res.ok) throw new Error(res.status === 404 ? '文档不存在或已删除' : await errorText(res));
  const version = Number(res.headers.get('X-Doc-Version'));
  return { text: await res.text(), version: Number.isInteger(version) && version > 0 ? version : 1 };
}

export async function sendFeedback(sessionId: string, messageId: string, rating: 1 | -1, reason?: string): Promise<void> {
  const res = await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Session-ID': sessionId },
    body: JSON.stringify({ message_id: messageId, rating, ...(reason ? { reason } : {}) }),
  });
  if (!res.ok) throw new Error(await errorText(res));
}

// 就绪：200 为 ok；503 带分项；接口不存在（旧进程）或网络错误按未知处理，不拦提问。
export async function checkReady(): Promise<'ok' | 'not_ready' | 'unknown'> {
  try {
    const res = await fetch('/api/health/ready');
    if (res.ok) return 'ok';
    if (res.status === 503) return 'not_ready';
  } catch {
    // 网络错误
  }
  return 'unknown';
}

// 409 是可恢复的用户决策，不能像普通 HTTP 错误一样丢掉冲突详情。
export class UploadError extends Error {
  constructor(public status: number, message: string, public conflicts: UploadConflict[] = []) {
    super(message); this.name = 'UploadError';
  }
}
async function uploadError(res: Response): Promise<UploadError> {
  const data = await res.json().catch(() => ({}));
  const detail = data.detail ?? data;
  const raw = Array.isArray(detail.conflicts) ? detail.conflicts : (Array.isArray(data.conflicts) ? data.conflicts : []);
  const conflicts: UploadConflict[] = raw.map((c: { name?: string; filename?: string; title?: string; version: number }) => ({ filename: c.name ?? c.filename ?? '', title: c.title ?? '', version: c.version }));
  const fallback = res.status === 403 ? '上传被拒绝，请从 localhost 或 127.0.0.1 同源页面访问' : res.status === 400 ? '文件不符合上传要求' : `HTTP ${res.status}`;
  return new UploadError(res.status, typeof detail === 'string' ? detail : (detail.message ?? fallback), conflicts);
}
export async function uploadFiles(files: File[], overwrite = false, signal?: AbortSignal): Promise<UploadJob> {
  const error = validateFiles(files);
  if (error) throw new UploadError(400, error);
  const body = new FormData();
  if (overwrite) body.append('overwrite', 'true');
  // 先按同一规则清洗，避免浏览器把双引号/换行编码成 %22/%0A，导致服务端名称和冲突映射不一致。
  files.forEach((file) => body.append('files', file, cleanFilename(file.name)));
  const res = await fetch('/api/uploads', {
    method: 'POST', headers: { [uploadConfig.clientHeader]: 'web' }, body, signal,
  });
  if (res.status !== 202) throw await uploadError(res);
  const job = await res.json();
  return normalizeJob({ status: 'queued', ...job });
}
export async function getJob(jobId: string, signal?: AbortSignal): Promise<UploadJob> {
  const res = await fetch(`/api/uploads/${encodeURIComponent(jobId)}`, { signal, headers: { [uploadConfig.clientHeader]: 'web' } });
  if (!res.ok) throw await uploadError(res);
  return normalizeJob(await res.json());
}
export async function listJobs(signal?: AbortSignal): Promise<UploadJob[]> {
  const res = await fetch('/api/uploads', { signal, headers: { [uploadConfig.clientHeader]: 'web' } });
  if (!res.ok) throw await uploadError(res);
  const data = await res.json();
  return (Array.isArray(data) ? data : (data.jobs ?? [])).map(normalizeJob);
}
export async function retryJob(jobId: string): Promise<UploadJob> {
  const res = await fetch(`/api/uploads/${encodeURIComponent(jobId)}/retry`, {
    method: 'POST', headers: { [uploadConfig.clientHeader]: 'web' },
  });
  if (res.status !== 202) throw await uploadError(res);
  return normalizeJob({ status: 'queued', ...await res.json() });
}

// 服务端持久化结构为 name/result；UI 中统一为 filename/status，空 result 是待处理而非失败。
function normalizeJob(job: Omit<UploadJob, 'files'> & { files?: (Partial<UploadFileResult> & { name?: string; result?: string | null })[] }): UploadJob {
  return { ...job, files: job.files?.map((f) => ({ ...f, filename: f.name ?? f.filename ?? '', status: f.result ?? f.status ?? '待处理' })) };
}
