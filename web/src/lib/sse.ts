// SSE 帧切分与 /api/chat 流式读取。sse.py 每帧一行 `data: {json}`。
import type { Filters } from './types';

export interface ChatEvent { type: string; [key: string]: unknown }

// 返回完整帧解析出的事件和剩余缓冲；坏帧丢弃，不中断整条流。
export function parseSSE(buffer: string): { events: ChatEvent[]; rest: string } {
  const events: ChatEvent[] = [];
  const frames = buffer.split('\n\n');
  const rest = frames.pop() ?? '';
  for (const frame of frames) {
    const data = frame.split('\n').filter((l) => l.startsWith('data:')).map((l) => l.slice(5).trimStart()).join('\n');
    if (!data) continue;
    try {
      const ev = JSON.parse(data);
      if (ev && typeof ev.type === 'string') events.push(ev);
    } catch {
      // 坏帧丢弃
    }
  }
  return { events, rest };
}

export async function errorText(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data.detail === 'string') return data.detail;
    if (Array.isArray(data.detail)) return data.detail.map((d: { msg?: string }) => d.msg).join('；');
  } catch {
    // 非 JSON 错误体
  }
  return `HTTP ${res.status}`;
}

export interface ChatRequest { question: string; sessionId: string; filters: Filters | null; useScope: boolean }

export async function streamChat(req: ChatRequest, onEvent: (ev: ChatEvent) => void, signal: AbortSignal): Promise<void> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Session-ID': req.sessionId },
    body: JSON.stringify({
      question: req.question,
      ...(req.filters ? { filters: req.filters } : {}),
      ...(req.useScope ? {} : { use_scope: false }),
    }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(await errorText(res));
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const parsed = parseSSE(done ? buffer + '\n\n' : buffer);
    buffer = parsed.rest;
    parsed.events.forEach(onEvent);
    if (done) break;
  }
}
