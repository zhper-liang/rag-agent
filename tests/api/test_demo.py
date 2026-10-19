# 前端只测后端这一侧：SPA 兜底路由、CSP、静态资源和未构建时的提示。
# 渲染器（转义、UTF-16 引用、补充说明、公式）的用例在 web/src/lib/*.test.ts，用 `make test-web` 跑。
import pytest

import ragagent.api.app as app_module
from tests.integration.helpers import KEYS, api_client, config, deps


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in KEYS:
        monkeypatch.setenv(name, 'test')


@pytest.fixture
def built(tmp_path, monkeypatch):
    # 假的构建产物，不依赖 node 和 npm run build。
    static = tmp_path / 'static'
    (static / 'assets').mkdir(parents=True)
    (static / 'index.html').write_text('<!doctype html><script type="module" src="/static/assets/app.js"></script>')
    (static / 'assets' / 'app.js').write_text('export {};')
    (static / 'assets' / 'app.css').write_text('body{}')
    (static / 'favicon.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    monkeypatch.setattr(app_module, 'STATIC', static)
    return static


def test_spa_routes_serve_index_with_strict_csp(tmp_path, embed_url, built):
    cfg = config(tmp_path)
    with api_client(cfg, deps(cfg, embed_url)) as client:
        for path in ('/', '/c/web-0123abcd', '/d/0123456789abcdef'):
            page = client.get(path)
            assert page.status_code == 200 and page.headers['content-type'].startswith('text/html'), path
            csp = page.headers['content-security-policy']
            assert "default-src 'none'" in csp and "script-src 'self'" in csp and "style-src 'self'" in csp
            assert 'unsafe-inline' not in csp and 'unsafe-eval' not in csp
            assert page.headers['cache-control'] == 'no-cache'
        # 非法会话 / 文档标识不兜底，避免任意路径都返回页面。
        for path in ('/c/bad%20id', '/c/' + 'a' * 129, '/d/0123456789ABCDEF', '/d/xyz', '/x/anything'):
            assert client.get(path).status_code == 404, path


def test_static_assets_types_and_traversal(tmp_path, embed_url, built):
    cfg = config(tmp_path)
    with api_client(cfg, deps(cfg, embed_url)) as client:
        assert 'javascript' in client.get('/static/assets/app.js').headers['content-type']
        assert client.get('/static/assets/app.css').headers['content-type'].startswith('text/css')
        assert client.get('/static/favicon.svg').headers['content-type'].startswith('image/svg+xml')
        assert client.get('/static/%2e%2e/app.py').status_code == 404
        assert client.get('/static/nope.js').status_code == 404


def test_unbuilt_frontend_returns_503_and_api_still_works(tmp_path, embed_url, monkeypatch):
    monkeypatch.setattr(app_module, 'STATIC', tmp_path / 'missing')
    cfg = config(tmp_path)
    with api_client(cfg, deps(cfg, embed_url)) as client:
        page = client.get('/')
        assert page.status_code == 503 and 'npm run build' in page.text
        assert page.headers['content-type'].startswith('text/plain')
        assert client.get('/c/web-1').status_code == 503
        assert client.get('/static/assets/app.js').status_code == 404
        assert client.get('/api/health/live').json() == {'status': 'ok'}
