import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

// 构建产物直接写进 API 的 static 目录（总揽 §4.3，D-020），不提交。
// base=/static/：index.html 由后端在 / /c/* /d/* 返回，资源都从 /static/ 取。
const outDir = fileURLToPath(new URL('../src/ragagent/api/static', import.meta.url));

export default defineConfig({
  base: '/static/',
  plugins: [react()],
  // 单包约 670 kB（highlight.js common + KaTeX）；只在本机同源加载，不拆包。
  build: { outDir, emptyOutDir: true, assetsDir: 'assets', sourcemap: false, chunkSizeWarningLimit: 800 },
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
});
