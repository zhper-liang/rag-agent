// 与后端 SSE 事件、/api/docs 返回对齐（后处理 §9、总揽 §4.3）。
export interface Citation { n: number; start: number; end: number; state: string }
export interface ImageRef { url: string; caption?: string | null }
export interface Source {
  cite: number; title?: string; section?: string | null; pages?: string | null;
  doc_id?: string; images?: ImageRef[];
}
export interface Final {
  type?: 'final';
  message_id?: string;
  citations?: Citation[];
  sources?: Source[];
  supplement?: { start: number | null } | null;
  notice?: string | null;
}
export interface Meta { scope_note: string | null; scoped: boolean; attachments: ImageRef[] }
export interface Filters { file_types?: string[]; doc_ids?: string[] }
export interface Doc { doc_id: string; title: string; file_type: string; pages: number | null; version: number; indexed_at?: string | null }

// 一轮问答在 localStorage 里的记录；字段沿用旧演示页（ragagent.sessions.v1），旧会话可直接打开。
export interface TurnRecord {
  question: string;
  filters: Filters | null;
  label: string;
  use_scope: boolean;
  answer: string;
  meta: Meta | null;
  final: Final | null;
  elapsed: number | null;
  rating?: 1 | -1;
  time_from?: string;
  time_to?: string;
}
export interface Session {
  id: string; title: string; turns: TurnRecord[]; updated: number;
  pinned?: boolean; renamed?: boolean;
}

export interface TimeRange { time_from?: string; time_to?: string }
export type JobStatus = 'queued' | 'cleaning' | 'chunking' | 'indexing' | 'done' | 'failed' | 'interrupted';
export interface UploadFileResult {
  filename: string; status: 'created' | 'updated' | 'unchanged' | 'duplicate' | 'failed' | string;
  reason?: string | null; doc_id?: string; version?: number;
}
export interface UploadJob { job_id: string; status: JobStatus; files?: UploadFileResult[]; error?: string | null }
export interface UploadConflict { filename: string; title: string; version: number }
