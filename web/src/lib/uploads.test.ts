import { describe, expect, it } from 'vitest';
import { resolveConflicts } from '../components/Uploads';
import { activeJob, cleanFilename, uploadAccept, uploadConfig, validateFiles } from './uploads';
import { scopeLabel } from './render';
import { parseSessions } from './sessions';

const f = (name: string, content = 'x') => new File([content], name);
describe('共享配置与上传校验', () => {
  it('accept 与 parse_routes 所有后缀一致', () => {
    expect(uploadAccept.split(',')).toEqual(uploadConfig.extensions.map((ext) => `.${ext}`));
    expect(uploadAccept).toContain('.markdown');
    expect(uploadAccept).not.toContain('.xlsx');
  });
  it('数量、大小上限在请求前检查，边界值可用', () => {
    const limit = uploadConfig.maxFileMb * 1024 * 1024;
    expect(validateFiles([{ name: 'a.pdf', size: limit }])).toBeNull();
    expect(validateFiles([{ name: 'a.pdf', size: limit + 1 }])).toContain('超过');
    expect(validateFiles(Array.from({ length: uploadConfig.maxFiles + 1 }, (_, i) => f(`${i}.md`)))).toContain('最多');
    expect(validateFiles([])).toBeTruthy();
  });
  it('清洗后重名、NFC、大小写后缀、路径、控制字符与截断保后缀', () => {
    expect(cleanFilename('C:\\fakepath\\  文\u0000 档.MD ')).toBe('文 档.md');
    expect(cleanFilename('a'.repeat(uploadConfig.maxNameChars + 3) + '.PDF')).toBe('a'.repeat(uploadConfig.maxNameChars) + '.pdf');
    expect(cleanFilename('..e\u0301.md')).toBe('é.md');
    expect(validateFiles([f('a?.MD'), f('a.md')])).toContain('重复');
    expect(validateFiles([f(' .md')])).toBeTruthy();
    expect(validateFiles([f('无后缀')])).toBeTruthy();
  });
  it('只有非终态需要轮询', () => {
    expect(['queued', 'cleaning', 'chunking', 'indexing'].every(activeJob)).toBe(true);
    expect(['done', 'failed', 'interrupted'].some(activeJob)).toBe(false);
  });
});
describe('逐个冲突决策', () => {
  const conflicts = ['a.md', 'b.md', 'c.md'].map((filename) => ({ filename, title: filename, version: 2 }));
  it('覆盖单独提交；改名保留原始内容；跳过不重传；非冲突文件不丢', async () => {
    const batches = resolveConflicts([f('a.md'), f('b.md', 'original'), f('c.md'), f('new.pdf')], conflicts,
      [{ action: 'overwrite', name: '' }, { action: 'rename', name: 'renamed.md' }, { action: 'skip', name: '' }]);
    expect(batches.map((b) => [b.overwrite, b.files.map((f) => f.name)])).toEqual([[false, ['renamed.md', 'new.pdf']], [true, ['a.md']]]);
    expect(await batches[0].files[0].text()).toBe('original');
  });
  it('全部跳过不提交空任务；同名、空名、改名后重名均阻止', () => {
    expect(resolveConflicts([f('a.md')], [conflicts[0]], [{ action: 'skip', name: '' }])).toEqual([]);
    expect(() => resolveConflicts([f('a.md')], [conflicts[0]], [{ action: 'rename', name: 'a.md' }])).toThrow('不同');
    expect(() => resolveConflicts([f('a.md')], [conflicts[0]], [{ action: 'rename', name: '' }])).toThrow('不同');
    expect(() => resolveConflicts([f('a.md'), f('new.md')], [conflicts[0]], [{ action: 'rename', name: 'new.md' }])).toThrow('重复');
  });
});
describe('日期标签与历史兼容', () => {
  it('只有时间仍显示范围且标明半开区间', () => {
    expect(scopeLabel(null, {}, { time_from: '2026-10-01', time_to: '2026-10-21' })).toBe('检索范围：入库时间 [2026-10-01, 2026-10-21)');
    expect(scopeLabel(null, {}, { time_to: '2026-10-21' })).toContain('[不限, 2026-10-21)');
  });
  it('旧会话照常打开，新会话保留可选日期', () => {
    const old = { question: 'q', answer: 'a', filters: null, label: '', use_scope: true, meta: null, final: {}, elapsed: 1 };
    const records = parseSessions(JSON.stringify([{ id: 'web-a', title: '', updated: 1, turns: [old, { ...old, time_from: '2026-10-01' }] }]))[0].turns;
    expect(records[0].time_from).toBeUndefined();
    expect(records[1].time_from).toBe('2026-10-01');
  });
});
