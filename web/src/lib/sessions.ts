// 会话历史只存 localStorage（总揽 §4.3）。键沿用旧演示页，旧会话可直接打开。
import type { Session, TurnRecord } from './types';

export const STORE = 'ragagent.sessions.v1';
const SESSION_ID = /^[A-Za-z0-9_-]{1,128}$/;

export const isSessionId = (s: unknown): s is string => SESSION_ID.test(String(s ?? ''));

export function newSessionId(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return 'web-' + Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

export function parseSessions(raw: string | null): Session[] {
  try {
    const data = JSON.parse(raw ?? '[]');
    if (!Array.isArray(data)) return [];
    return data.filter((s) => s && isSessionId(s.id) && Array.isArray(s.turns))
      .map((s) => ({ ...s, title: String(s.title ?? ''), updated: Number(s.updated) || 0 }));
  } catch {
    return [];
  }
}

export function loadSessions(storage: Storage = localStorage): Session[] {
  return parseSessions(storage.getItem(STORE));
}

// 超出配额时从最旧的非当前会话开始丢；返回实际保存下来的列表。
export function saveSessions(list: Session[], keepId: string | null, storage: Storage = localStorage): { saved: Session[]; full: boolean } {
  const saved = [...list].sort((a, b) => b.updated - a.updated);
  for (;;) {
    try {
      storage.setItem(STORE, JSON.stringify(saved));
      return { saved, full: false };
    } catch {
      const idx = saved.findLastIndex((s) => s.id !== keepId);
      if (idx < 0) return { saved, full: true };
      saved.splice(idx, 1);
    }
  }
}

export function appendTurn(list: Session[], id: string, rec: TurnRecord, now = Date.now()): Session[] {
  const found = list.find((s) => s.id === id);
  if (!found) return [...list, { id, title: rec.question.slice(0, 40), turns: [rec], updated: now }];
  return list.map((s) => (s.id === id ? { ...s, turns: [...s.turns, rec], updated: now } : s));
}

export function patchSession(list: Session[], id: string, patch: Partial<Session>): Session[] {
  return list.map((s) => (s.id === id ? { ...s, ...patch } : s));
}

export interface Group { label: string; sessions: Session[] }

// 置顶在最前；其余按本地日期分：今天 / 昨天 / 7 天内 / 30 天内 / 更早。
export function groupSessions(list: Session[], query = '', now = new Date()): Group[] {
  const q = query.trim().toLowerCase();
  const shown = list.filter((s) => !q || s.title.toLowerCase().includes(q)).sort((a, b) => b.updated - a.updated);
  const day = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const DAY = 86_400_000;
  const buckets: [string, (t: number) => boolean][] = [
    ['今天', (t) => t >= day],
    ['昨天', (t) => t >= day - DAY],
    ['7 天内', (t) => t >= day - 6 * DAY],
    ['30 天内', (t) => t >= day - 29 * DAY],
    ['更早', () => true],
  ];
  const groups: Group[] = [{ label: '置顶', sessions: shown.filter((s) => s.pinned) }];
  const rest = shown.filter((s) => !s.pinned);
  const used = new Set<string>();
  for (const [label, test] of buckets) {
    const items = rest.filter((s) => !used.has(s.id) && test(s.updated));
    items.forEach((s) => used.add(s.id));
    groups.push({ label, sessions: items });
  }
  return groups.filter((g) => g.sessions.length);
}
