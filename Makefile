.PHONY: test test-web

test:
	uv run pytest

# 前端渲染器与会话/路由单测（web/，vitest）；需先在 web/ 下 npm ci
test-web:
	cd web && npm test
