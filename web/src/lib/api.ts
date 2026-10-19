// 其余同源接口：文档列表、原文、反馈、就绪检查。
import { errorText } from './sse';
import { isDocId } from './render';
import type { Doc } from './types';

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
