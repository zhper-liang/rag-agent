// 执行：playwright-cli -s=ragagent-slice2 run-code --filename=web/tests/uploads-errors.browser.js
// production preview :4173，验证前置拒绝、403 错误不被轮询抹掉、重试提交。
async (page) => {
  const assert = (c, m) => { if (!c) throw new Error(m); };
  let calls = 0, refreshed = false, quoteConflict = false;
  await page.route('**/api/health/ready', (r) => r.fulfill({ json: {} }));
  await page.route('**/api/uploads', async (route) => {
    if (route.request().method() === 'GET') return route.fulfill({ json: { jobs: [{ job_id: 'active', status: refreshed ? 'done' : 'queued', files: [{ name: 'pending.md', result: null }] }] } });
    calls++;
    if (quoteConflict) {
      const body = route.request().postData() ?? '';
      assert(body.includes('filename="ab.md"') && !body.includes('%22'), 'filename not sanitized before multipart encoding');
      return route.fulfill({ status: 409, json: { conflicts: [{ name: 'ab.md', title: '已有引号文件', version: 2 }] } });
    }
    return route.fulfill(calls === 1 ? { status: 403, json: { detail: '测试同源拒绝' } } : { status: 202, json: { job_id: 'ok', files: [{ name: 'retry.md' }] } });
  });
  await page.route('**/api/uploads/*', (route) => { refreshed = true; return route.fulfill({ json: { job_id: 'active', status: 'done', files: [{ name: 'pending.md', result: 'created' }] } }); });
  await page.goto('http://127.0.0.1:4173/static/');
  await page.getByRole('region', { name: '上传任务' }).waitFor();
  const input = page.getByLabel('选择上传文件');
  const limits = await page.getByRole('button', { name: '上传文档', exact: true }).getAttribute('title');
  const maxFiles = Number(limits.match(/最多 (\d+) 个/)[1]);
  const maxMb = Number(limits.match(/单个 (\d+) MB/)[1]);
  await input.evaluate((el, count) => {
    const data = new DataTransfer();
    for (let i = 0; i <= count; i++) data.items.add(new File(['x'], `${i}.md`));
    el.files = data.files; el.dispatchEvent(new Event('change', { bubbles: true }));
  }, maxFiles);
  await page.getByText(`一次最多上传 ${maxFiles} 个文件`, { exact: false }).waitFor();
  assert(calls === 0, 'excess file count reached network');
  await input.evaluate((el, mb) => {
    const data = new DataTransfer(); data.items.add(new File([new Uint8Array(mb * 1024 * 1024 + 1)], 'large.md'));
    el.files = data.files; el.dispatchEvent(new Event('change', { bubbles: true }));
  }, maxMb);
  await page.getByText(`large.md 超过单个文件 ${maxMb} MB 的限制`, { exact: false }).waitFor();
  assert(calls === 0, 'oversized file reached network');
  await input.evaluate((el) => {
    const data = new DataTransfer(); data.items.add(new File(['x'], 'bad.xlsx'));
    el.files = data.files; el.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await page.getByText('不支持的文件类型：bad.xlsx', { exact: false }).waitFor();
  assert(calls === 0, 'invalid file sent request');
  await input.evaluate((el) => {
    const data = new DataTransfer(); data.items.add(new File(['x'], 'retry.md'));
    el.files = data.files; el.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await page.getByText('测试同源拒绝', { exact: false }).waitFor();
  await page.waitForTimeout(2300);
  assert((await page.getByText('测试同源拒绝', { exact: false }).count()) === 1, 'successful poll cleared submission error');
  await page.getByRole('button', { name: '重试提交', exact: true }).click();
  await page.waitForFunction(() => !document.querySelector('button[aria-label="上传文档"]').disabled);
  assert(calls === 2, 'retry not submitted exactly once');
  quoteConflict = true;
  await input.evaluate((el) => {
    const data = new DataTransfer(); data.items.add(new File(['x'], 'a"b.MD'));
    el.files = data.files; el.dispatchEvent(new Event('change', { bubbles: true }));
  });
  const dialog = page.getByRole('dialog', { name: '处理同名文件' });
  await dialog.waitFor();
  await dialog.getByText('已有引号文件 · 现有版本 v2', { exact: true }).waitFor();
  await dialog.getByLabel('ab.md 的处理方式').selectOption('skip');
  await dialog.getByRole('button', { name: '确认并继续' }).click();
  await page.waitForFunction(() => !document.querySelector('.conflict-dialog') && !document.querySelector('button[aria-label="上传文档"]').disabled);
  assert(calls === 3, 'skip failed to match sanitized filename and resubmitted conflict');
  return { frontValidationNoNetwork: true, countAndSizeNoNetwork: true, submissionErrorSurvivedPoll: true, retrySubmission: true, quotedFilenameConflictSkip: true };
}
