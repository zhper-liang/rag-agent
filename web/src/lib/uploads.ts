// 参数在 Vite 构建/开发时从 config.yaml 提取，不在浏览器端维护第二套值。
export interface UploadConfig { extensions: string[]; maxFiles: number; maxFileMb: number; maxNameChars: number; clientHeader: string }
declare const __UPLOAD_CONFIG__: UploadConfig;
export const uploadConfig = __UPLOAD_CONFIG__;
export const uploadAccept = uploadConfig.extensions.map((ext) => `.${ext}`).join(',');
export function validateFiles(files: Pick<File, 'name' | 'size'>[]): string | null {
  if (!files.length) return '请选择文件';
  if (files.length > uploadConfig.maxFiles) return `一次最多上传 ${uploadConfig.maxFiles} 个文件`;
  const names = new Set<string>();
  for (const file of files) {
    const name = cleanFilename(file.name);
    if (!name || name.startsWith('.')) return `文件名无效：${file.name}`;
    const ext = name.split('.').pop()?.toLowerCase();
    if (!name.includes('.') || !ext || !uploadConfig.extensions.includes(ext)) return `不支持的文件类型：${file.name}`;
    if (file.size > uploadConfig.maxFileMb * 1024 * 1024) return `${file.name} 超过单个文件 ${uploadConfig.maxFileMb} MB 的限制`;
    if (names.has(name)) return `清洗后文件名重复：${name}，请先改名`;
    names.add(name);
  }
  return null;
}
// 用于把服务器返回的清洗后名称对应回原 File，以及提前发现请求内重名。
export function cleanFilename(raw: string): string {
  const base = raw.split(/[\\/]/).pop()!.normalize('NFC')
    .replace(/[\p{C}/\\:*?"<>|]/gu, '').replace(/\s+/gu, ' ').replace(/^[ .]+|[ .]+$/g, '');
  const dot = base.lastIndexOf('.');
  if (dot < 0) return base;
  const stem = [...base.slice(0, dot).replace(/^[ .]+|[ .]+$/g, '')].slice(0, uploadConfig.maxNameChars).join('').replace(/[ .]+$/g, '');
  return stem ? stem + base.slice(dot).toLowerCase() : '';
}
export const activeJob = (status: string) => ['queued', 'cleaning', 'chunking', 'indexing'].includes(status);
