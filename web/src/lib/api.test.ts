import { afterEach, describe, expect, it, vi } from 'vitest';
import { getJob, listJobs, retryJob, uploadFiles, UploadError } from './api';
import { uploadConfig } from './uploads';
import { streamChat } from './sse';

const file = () => new File(['content'], '文档.md', { type: 'text/markdown' });
afterEach(() => vi.unstubAllGlobals());
const mock = (body: unknown, status = 200) => {
  const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status }));
  vi.stubGlobal('fetch', fetch); return fetch;
};
describe('上传 API 契约', () => {
  it('202 带自定义头，不手设 multipart Content-Type；覆盖通过表单字段提交', async () => {
    const fetch = mock({ job_id: 'job-1', files: [{ name: '文档.md' }] }, 202);
    expect(await uploadFiles([file()], true)).toMatchObject({ job_id: 'job-1', status: 'queued', files: [{ filename: '文档.md', status: '待处理' }] });
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe('/api/uploads');
    expect(init.headers).toEqual({ [uploadConfig.clientHeader]: 'web' });
    expect(init.body.get('overwrite')).toBe('true');
    expect(init.body.getAll('files')).toHaveLength(1);
  });
  it('传输清洗后的文件名，避免双引号被 multipart 编为 %22 后冲突无法匹配', async () => {
    const fetch = mock({ job_id: 'quoted', files: [{ name: 'ab.md' }] }, 202);
    const original = new File(['内容不变'], 'a"b.MD', { type: 'text/markdown' });
    await uploadFiles([original]);
    const sent = fetch.mock.calls[0][1].body.get('files') as File;
    expect(sent.name).toBe('ab.md');
    expect(await sent.text()).toBe('内容不变');
    expect(original.name).toBe('a"b.MD');
  });
  it.each([
    { conflicts: [{ name: '文档.md', title: '已有标题', version: 3 }] },
    { detail: { conflicts: [{ name: '文档.md', title: '已有标题', version: 3 }] } },
  ])('409 保留文件名、标题与现有版本：%j', async (body) => {
    mock(body, 409);
    await expect(uploadFiles([file()])).rejects.toMatchObject({ status: 409, conflicts: [{ filename: '文档.md', title: '已有标题', version: 3 }] });
  });
  it.each([400, 403])('%i 与 409 区分并保留可读原因', async (status) => {
    mock({ detail: '服务端拒绝原因' }, status);
    await expect(uploadFiles([file()])).rejects.toMatchObject({ status, message: '服务端拒绝原因', conflicts: [] });
  });
  it('非 JSON 403 使用安全友好提示；意外 200 不当成成功', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('Forbidden', { status: 403 })));
    await expect(uploadFiles([file()])).rejects.toMatchObject({ status: 403, message: expect.stringContaining('同源') });
    mock({ job_id: 'bad' }, 200);
    await expect(uploadFiles([file()])).rejects.toBeInstanceOf(UploadError);
  });
  it('查询/恢复/重试都带头，任务结果映射 name/result/reason', async () => {
    const raw = { job_id: 'job-1', status: 'interrupted', files: [{ name: '文档.md', result: 'failed', reason: '请重试' }] };
    let fetch = mock({ jobs: [raw] });
    expect((await listJobs())[0]).toMatchObject({ status: 'interrupted', files: [{ filename: '文档.md', status: 'failed', reason: '请重试' }] });
    expect(fetch.mock.calls[0][1].headers).toEqual({ [uploadConfig.clientHeader]: 'web' });
    fetch = mock(raw);
    expect(await getJob('job-1')).toMatchObject({ job_id: 'job-1' });
    expect(fetch.mock.calls[0][0]).toBe('/api/uploads/job-1');
    expect(fetch.mock.calls[0][1].headers).toEqual({ [uploadConfig.clientHeader]: 'web' });
    fetch = mock({ ...raw, status: 'queued' }, 202);
    expect(await retryJob('job-1')).toMatchObject({ status: 'queued' });
    expect(fetch.mock.calls[0]).toEqual(['/api/uploads/job-1/retry', { method: 'POST', headers: { [uploadConfig.clientHeader]: 'web' } }]);
  });
  it('前端文件验证失败时不发请求', async () => {
    const fetch = mock({});
    await expect(uploadFiles([new File(['x'], '禁止.xlsx')])).rejects.toMatchObject({ status: 400 });
    expect(fetch).not.toHaveBeenCalled();
  });
});
describe('时间范围请求', () => {
  it('时间独立于 filters，取消软范围仍保留时间参数', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response('data: {"type":"final"}\n\n', { status: 200 }));
    vi.stubGlobal('fetch', fetch);
    const onEvent = vi.fn();
    await streamChat({ question: 'q', sessionId: 'web-a', filters: { file_types: ['pdf'] }, useScope: false, time_from: '2026-10-01', time_to: '2026-10-21' }, onEvent, new AbortController().signal);
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ question: 'q', filters: { file_types: ['pdf'] }, use_scope: false, time_from: '2026-10-01', time_to: '2026-10-21' });
    expect(onEvent.mock.calls[0][0]).toEqual({ type: 'final' });
  });
  it('旧会话没有时间字段时不会发送空值', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response('data: {"type":"final"}\n\n'));
    vi.stubGlobal('fetch', fetch);
    await streamChat({ question: 'old', sessionId: 'web-a', filters: null, useScope: true }, () => {}, new AbortController().signal);
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ question: 'old' });
  });
});
