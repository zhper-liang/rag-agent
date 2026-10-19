import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { readFileSync } from 'node:fs';
import { parse } from 'yaml';
import { defineConfig } from 'vitest/config';

// 构建产物直接写进 API 的 static 目录（总揽 §4.3，D-020），不提交。
// base=/static/：index.html 由后端在 / /c/* /d/* 返回，资源都从 /static/ 取。
const outDir = fileURLToPath(new URL('../src/ragagent/api/static', import.meta.url));

// 与后端共用参数源；只注入上传所需公开字段，不能将含接入点/密钥变量名的配置整体暴露。
const config = parse(readFileSync(new URL('../config.yaml', import.meta.url), 'utf8'));
const upload = config.api.upload;
export default defineConfig({
  define: { __UPLOAD_CONFIG__: JSON.stringify({
    extensions: [...new Set(Object.values(config.ingest.parse_routes).flat())],
    maxFiles: upload.max_files, maxFileMb: upload.max_file_mb,
    maxNameChars: upload.max_name_chars, clientHeader: upload.client_header,
  }) },
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
