// 准备：mkdir -p /tmp/ragagent-slice2；在 web/ 启动 npm exec vite preview -- --host 127.0.0.1 --port 4173
// 执行：playwright-cli -s=ragagent-slice2 run-code --filename=web/tests/uploads.browser.js
// 使用 production preview :4173 + 与 api/uploads.py 相同的 wire 契约；不向真实语料写入文件。
async (page) => {
  const assert = (condition, message) => { if (!condition) throw new Error(message); };
  const consoleErrors = [], pageErrors = [], uploads = [], chats = [], jobGets = [];
  page.on('console', (message) => { if (message.type() === 'error') consoleErrors.push(message.text()); });
  page.on('pageerror', (error) => pageErrors.push(error.message));
  let docRequests = 0, docAvailable = false, next = 0, initialConflict = true;
  const jobs = new Map([
    ['recover', { job_id: 'recover', status: 'queued', files: [{ name: '恢复.md', result: null }] }],
    ['interrupted', { job_id: 'interrupted', status: 'interrupted', files: [{ name: '中断.pdf', result: 'failed', reason: '重启中断' }] }],
  ]);
  const polls = new Map();
  const json = (route, body, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
  await page.route('**/api/**', async (route) => {
    const req = route.request(), path = '/' + req.url().split('/').slice(3).join('/').split('?')[0];
    if (path === '/api/health/ready') return json(route, {});
    if (path === '/api/docs') {
      docRequests++;
      return json(route, { docs: docAvailable ? [{ doc_id: '0123456789abcdef', title: '入库新文档', file_type: 'md', version: 2, pages: null, indexed_at: '2026-10-20T01:00:00+00:00' }] : [], truncated: false });
    }
    if (path === '/api/chat') {
      chats.push(req.postDataJSON());
      return route.fulfill({ contentType: 'text/event-stream', body: [
        { type: 'meta', scope_note: '问题识别范围', scoped: true, attachments: [] },
        { type: 'token', text: '在所选入库时间内回答。' },
        { type: 'final', message_id: `message-${chats.length}`, sources: [], citations: [] },
      ].map((ev) => `data: ${JSON.stringify(ev)}\n\n`).join('') });
    }
    assert(req.headers()['x-rag-client'] === 'web', `missing client header: ${path}`);
    if (path === '/api/uploads' && req.method() === 'GET') return json(route, { jobs: [...jobs.values()] });
    if (path === '/api/uploads' && req.method() === 'POST') {
      const body = req.postData() ?? '';
      uploads.push({ body, url: req.url() });
      if (initialConflict) {
        initialConflict = false;
        return json(route, { conflicts: ['a.md', 'b.md', 'c.md'].map((name) => ({ name, title: `已有 ${name}`, version: 3 })) }, 409);
      }
      const names = [...body.matchAll(/filename="([^"]+)"/g)].map((m) => m[1]);
      const job = { job_id: `new-${++next}`, status: 'queued', files: names.map((name) => ({ name, result: null })) };
      jobs.set(job.job_id, job);
      return json(route, { job_id: job.job_id, files: job.files }, 202);
    }
    const retry = path.match(/^\/api\/uploads\/([^/]+)\/retry$/);
    if (retry) {
      const job = jobs.get(retry[1]); job.status = 'queued';
      return json(route, job, 202);
    }
    const detail = path.match(/^\/api\/uploads\/([^/]+)$/);
    if (detail) {
      jobGets.push({ id: detail[1], at: Date.now() });
      const job = jobs.get(detail[1]);
      const count = (polls.get(detail[1]) ?? 0) + 1; polls.set(detail[1], count);
      job.status = count === 1 ? 'indexing' : 'done';
      if (job.status === 'done') { job.files.forEach((f) => f.result = 'created'); docAvailable = true; }
      return json(route, job);
    }
    throw new Error(`unexpected request ${req.method()} ${path}`);
  });
  // 对生产 HTML 施加与后端相同的 CSP，不能只在 Vite 的宽松环境里测试。
  const serveHtml = async (route) => {
    const response = await route.fetch({ url: 'http://127.0.0.1:4173/static/' });
    await route.fulfill({ response, headers: { ...response.headers(),
      'content-security-policy': "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'" } });
  };
  await page.route('**/static/', serveHtml);
  // Vite preview 只服务 /static/；模拟 API 已有的 /c/* SPA fallback。
  await page.route('**/c/**', serveHtml);
  await page.goto('http://127.0.0.1:4173/static/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByRole('region', { name: '上传任务' }).waitFor();
  await page.getByRole('button', { name: '检索范围', exact: true }).click();
  await page.getByText('还没有已入库的文档').waitFor();
  const before = docRequests;
  // 点击任务重试会正常关闭范围弹层，完成后重新打开时应加载新列表。
  await page.getByRole('button', { name: '重试', exact: true }).click();
  await page.getByText('上传任务已完成，文档列表已刷新').waitFor();
  await page.getByRole('button', { name: '检索范围', exact: true }).click();
  await page.getByText('入库新文档', { exact: true }).waitFor();
  assert(docRequests > before, 'done did not refresh open doc list');
  assert((await page.locator('.doc-list small').filter({ hasText: '入库：2026-10-20' }).count()) === 1, 'missing indexed date');
  assert(polls.get('recover') >= 2 && polls.get('interrupted') >= 2, 'restore/retry not polled');
  const times = jobGets.filter((x) => x.id === 'recover');
  assert(times[1].at - times[0].at >= 1800, 'polling faster than 2s');
  const requestsAtDone = jobGets.length;
  await page.waitForTimeout(2300);
  assert(jobGets.length === requestsAtDone, 'terminal job kept polling');
  await page.getByRole('button', { name: '检索范围', exact: true }).click();
  const input = page.getByLabel('选择上传文件');
  await input.evaluate((el) => {
    const transfer = new DataTransfer();
    for (const name of ['a.md', 'b.md', 'c.md', 'new.pdf']) transfer.items.add(new File(['updated'], name, { type: name.endsWith('.pdf') ? 'application/pdf' : 'text/markdown' }));
    el.files = transfer.files;
    el.dispatchEvent(new Event('change', { bubbles: true }));
  });
  const dialog = page.getByRole('dialog', { name: '处理同名文件' });
  await dialog.waitFor();
  assert((await dialog.getByText('已有 a.md · 现有版本 v3', { exact: true }).count()) === 1, 'missing conflict version');
  await dialog.getByLabel('a.md 的处理方式').selectOption('overwrite');
  await dialog.getByLabel('b.md 的处理方式').selectOption('rename');
  await dialog.getByLabel('b.md 的新文件名').fill('renamed.md');
  await dialog.getByRole('button', { name: '确认并继续' }).click();
  await page.waitForFunction(() => !document.querySelector('.conflict-dialog'));
  await page.waitForFunction(() => !document.querySelector('button[aria-label="上传文档"]').disabled);
  assert(uploads.length === 3, `expected conflict + 2 batches, got ${uploads.length}`);
  assert(uploads[1].body.includes('filename="renamed.md"') && uploads[1].body.includes('filename="new.pdf"'), 'normal/rename batch missing files');
  assert(!uploads[1].body.includes('name="overwrite"'), 'normal batch accidentally overwrites');
  assert(uploads[2].body.includes('name="overwrite"\r\n\r\ntrue') && uploads[2].body.includes('filename="a.md"'), 'overwrite must be multipart');
  assert(!uploads.slice(1).some((u) => u.body.includes('filename="c.md"')), 'skipped file resubmitted');
  await page.getByRole('button', { name: '检索范围', exact: true }).click();
  await page.getByLabel('入库开始日期').fill('2026-10-01');
  await page.getByLabel('入库结束日期').fill('2026-10-21');
  await page.getByRole('textbox', { name: '问题', exact: true }).fill('按日期查资料');
  await page.getByRole('button', { name: '发送', exact: true }).click();
  await page.getByRole('button', { name: '重新生成', exact: true }).waitFor();
  assert(chats[0].time_from === '2026-10-01' && chats[0].time_to === '2026-10-21', 'date range not sent');
  assert((await page.getByText('检索范围：入库时间 [2026-10-01, 2026-10-21)', { exact: true }).count()) === 1, 'scope label missing dates');
  // 新轮次导致输入框重挂载，日期仍在 App 的状态中；对话历史保存同一日期。
  await page.getByRole('button', { name: '重新生成', exact: true }).click();
  await page.waitForFunction(() => document.querySelectorAll('button[aria-label="重新生成"]').length === 2);
  await page.getByRole('button', { name: '取消限定，在全部文档中重新回答' }).first().click();
  await page.waitForFunction(() => document.querySelectorAll('button[aria-label="重新生成"]').length === 3);
  assert(chats.length === 3 && chats.every((c) => c.time_from === '2026-10-01' && c.time_to === '2026-10-21'), 'regenerate/unscope lost dates');
  assert(chats[2].use_scope === false, 'unscope did not disable soft scope');
  const history = await page.evaluate(() => JSON.parse(localStorage.getItem('ragagent.sessions.v1')));
  assert(history[0].turns.length === 3 && history[0].turns.every((t) => t.time_to === '2026-10-21'), 'date persistence failed');
  await page.reload();
  await page.getByRole('button', { name: '重新生成', exact: true }).nth(2).waitFor();
  await page.getByRole('button', { name: '检索范围', exact: true }).click();
  await page.getByLabel('入库开始日期').fill('2026-10-21');
  await page.getByLabel('入库结束日期').fill('2026-10-20');
  await page.getByRole('textbox', { name: '问题', exact: true }).fill('bad dates');
  assert(await page.getByRole('button', { name: '发送', exact: true }).isDisabled(), 'reversed dates accepted');
  await page.getByRole('button', { name: '清空范围', exact: true }).click();
  assert(await page.getByRole('button', { name: '发送', exact: true }).isEnabled(), 'clearing dates did not reset validation');
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.screenshot({ path: '/tmp/ragagent-slice2/dark-mobile.png', fullPage: true });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'mobile horizontal overflow');
  assert(pageErrors.length === 0, `runtime errors: ${pageErrors.join(';')}`);
  assert(!consoleErrors.some((e) => /Content Security Policy|Refused to|violates/i.test(e)), `CSP errors: ${consoleErrors.join(';')}`);
  return { uploads: uploads.length, chatRequests: chats.length, docRequests, polls: Object.fromEntries(polls), consoleErrors, pageErrors, screenshot: '/tmp/ragagent-slice2/dark-mobile.png' };
}
