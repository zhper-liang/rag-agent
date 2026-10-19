// 执行：playwright-cli -s=ragagent-slice2 run-code --filename=web/tests/uploads-errors.browser.js
// production preview :4173，验证前置拒绝、403 错误不被轮询抹掉、重试提交。
async (page) => {
  const assert = (c, m) => { if (!c) throw new Error(m); };
  let calls = 0, refreshed = false;
  await page.route('**/api/health/ready', (r) => r.fulfill({ json: {} }));
  await page.route('**/api/uploads', async (route) => {
    if (route.request().method() === 'GET') return route.fulfill({ json: { jobs: [{ job_id: 'active', status: refreshed ? 'done' : 'queued', files: [{ name: 'pending.md', result: null }] }] } });
    calls++;
    return route.fulfill(calls === 1 ? { status: 403, json: { detail: '测试同源拒绝' } } : { status: 202, json: { job_id: 'ok', files: [{ name: 'retry.md' }] } });
  });
  await page.route('**/api/uploads/*', (route) => { refreshed = true; return route.fulfill({ json: { job_id: 'active', status: 'done', files: [{ name: 'pending.md', result: 'created' }] } }); });
  await page.goto('http://127.0.0.1:4173/static/');
  await page.getByRole('region', { name: '上传任务' }).waitFor();
  const input = page.getByLabel('选择上传文件');
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
  return { frontValidationNoNetwork: true, submissionErrorSurvivedPoll: true, retrySubmission: true };
}
