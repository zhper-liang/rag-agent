import { describe, expect, it } from 'vitest';
import { parseRoute, routePath } from './router';
import { appendTurn, groupSessions, parseSessions, saveSessions } from './sessions';
import type { Session, TurnRecord } from './types';

const rec = (question: string): TurnRecord => ({
  question, filters: null, label: '', use_scope: true, answer: 'a', meta: null, final: {}, elapsed: 1,
});

class Mem implements Storage {
  data = new Map<string, string>();
  limit = Infinity;
  get length() { return this.data.size; }
  clear() { this.data.clear(); }
  getItem(k: string) { return this.data.get(k) ?? null; }
  key(i: number) { return [...this.data.keys()][i] ?? null; }
  removeItem(k: string) { this.data.delete(k); }
  setItem(k: string, v: string) {
    if (v.length > this.limit) throw new DOMException('quota', 'QuotaExceededError');
    this.data.set(k, v);
  }
}

describe('会话存储', () => {
  it('读旧格式、丢掉非法条目', () => {
    const list = parseSessions(JSON.stringify([
      { id: 'web-1', title: '旧', turns: [rec('q')], updated: 5 },
      { id: '../x', title: 'bad', turns: [] },
      { id: 'web-2', title: 'no turns' },
    ]));
    expect(list.map((s) => s.id)).toEqual(['web-1']);
    expect(parseSessions('not json')).toEqual([]);
  });

  it('追加轮次：新会话取问题前 40 字做标题', () => {
    let list: Session[] = [];
    list = appendTurn(list, 'web-a', rec('x'.repeat(50)), 10);
    list = appendTurn(list, 'web-a', rec('y'), 20);
    expect(list).toHaveLength(1);
    expect(list[0]!.title).toHaveLength(40);
    expect(list[0]!.turns).toHaveLength(2);
    expect(list[0]!.updated).toBe(20);
  });

  it('超配额时从最旧的非当前会话开始丢', () => {
    const store = new Mem();
    const list: Session[] = [1, 2, 3].map((i) => ({ id: `web-${i}`, title: `t${i}`, turns: [rec('q'.repeat(100))], updated: i }));
    store.limit = JSON.stringify(list.slice(1)).length;
    const { saved, full } = saveSessions(list, 'web-1', store);
    expect(full).toBe(false);
    expect(saved.map((s) => s.id).sort()).toEqual(['web-1', 'web-3']);
  });

  it('分组：置顶在前，按日期分桶，可搜索', () => {
    const now = new Date(2026, 9, 19, 15, 0, 0);
    const at = (d: number) => new Date(2026, 9, d, 12).getTime();
    const list: Session[] = [
      { id: 'web-a', title: '今天的', turns: [], updated: at(19) },
      { id: 'web-b', title: '昨天的', turns: [], updated: at(18) },
      { id: 'web-c', title: '上周的', turns: [], updated: at(14) },
      { id: 'web-d', title: '很久前', turns: [], updated: at(1) - 30 * 86_400_000 },
      { id: 'web-e', title: '置顶的', turns: [], updated: at(1), pinned: true },
    ];
    const groups = groupSessions(list, '', now);
    expect(groups.map((g) => g.label)).toEqual(['置顶', '今天', '昨天', '7 天内', '更早']);
    expect(groupSessions(list, '昨天', now).flatMap((g) => g.sessions.map((s) => s.id))).toEqual(['web-b']);
  });
});

describe('路由', () => {
  it('解析与生成', () => {
    expect(parseRoute('/')).toEqual({ name: 'home' });
    expect(parseRoute('/c/web-abc')).toEqual({ name: 'chat', sessionId: 'web-abc' });
    expect(parseRoute('/c/bad%20id')).toEqual({ name: 'home' });
    expect(parseRoute('/d/0123456789abcdef', '?s=%E7%AC%AC2%E7%AB%A0')).toEqual({ name: 'doc', docId: '0123456789abcdef', section: '第2章' });
    expect(parseRoute('/d/xyz')).toEqual({ name: 'home' });
    expect(routePath({ name: 'doc', docId: '0123456789abcdef', section: 'a > b' })).toBe('/d/0123456789abcdef?s=a%20%3E%20b');
  });
});
