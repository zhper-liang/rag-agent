"""SQLite 增量登记（数据清洗 §12，问题预处理 §8.2）。

写入通过 RegistryDB 方法，documents 与 doc_titles 始终在同一事务更新。
FTS 的 filename 为无扩展名 basename（不是新增 documents 契约字段）。
消费方可直接使用 conn/get_connection() 进行只读 SQL。
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from pathlib import Path, PurePosixPath
import hashlib
import re
import sqlite3
import unicodedata

class IngestLocked(RuntimeError):
    pass


@contextmanager
def ingest_lock(path: str | Path):
    """入库独占锁（服务化部署 §8）：清洗与写索引命令互斥，拿不到立即失败，不排队。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise IngestLocked(f'已有入库任务在运行（锁文件 {path}）') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


STATUSES = ('pending','parsing','cleaned','indexed','empty','duplicate','unsupported','failed','deleted')
ID = re.compile(r'^[0-9a-f]{16}$')
FIELDS = {'source_path','title','file_type','parser','content_hash','version','status',
          'mineru_batch_id','pages','duplicate_of','error'}


def make_doc_id(relative_path: str) -> str:
    path = str(PurePosixPath(relative_path.replace('\\','/')))
    return hashlib.sha1(path.encode('utf-8')).hexdigest()[:16]


def _filename(path):
    return PurePosixPath(path.replace('\\','/')).stem


def normalize_title(value: str) -> str:
    value = unicodedata.normalize('NFKC',value).strip()
    value = re.sub(r'[《》〈〉"“”‘’\']','',value)
    value = re.sub(r'\.(?:pdf|docx?|pptx?|md|markdown|txt|png|jpe?g)$','',value,flags=re.I)
    return value.casefold().strip()


def connect_db(path: str | Path = 'data/registry.db', *, readonly=False) -> sqlite3.Connection:
    path = Path(path) if str(path) != ':memory:' else path
    if readonly:
        if str(path) == ':memory:':
            raise ValueError('内存库不能只读重新连接')
        conn = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True,timeout=30)
    else:
        if isinstance(path,Path):
            path.parent.mkdir(parents=True,exist_ok=True)
        conn = sqlite3.connect(str(path),timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys=ON')
    conn.execute('PRAGMA busy_timeout=30000')
    if not readonly:
        conn.execute('PRAGMA journal_mode=WAL')
        init_db(conn)
    return conn


def init_db(conn):
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY CHECK(length(doc_id)=16 AND doc_id NOT GLOB '*[^0-9a-f]*'),
        source_path TEXT NOT NULL UNIQUE,
        title TEXT NOT NULL,
        file_type TEXT NOT NULL,
        parser TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        version INTEGER NOT NULL CHECK(version>=1),
        status TEXT NOT NULL CHECK(status IN ('pending','parsing','cleaned','indexed','empty','duplicate','unsupported','failed','deleted')),
        mineru_batch_id TEXT,
        pages INTEGER CHECK(pages IS NULL OR pages>=0),
        duplicate_of TEXT,
        error TEXT,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS documents_status ON documents(status);
    CREATE INDEX IF NOT EXISTS documents_content_hash ON documents(content_hash);
    CREATE VIRTUAL TABLE IF NOT EXISTS doc_titles USING fts5(
        doc_id UNINDEXED,title,filename,tokenize='trigram'
    );
    ''')


class RegistryDB:
    def __init__(self,path: str | Path = 'data/registry.db', *, readonly=False):
        self.conn = connect_db(path,readonly=readonly)
        self.readonly = readonly

    def get_connection(self):
        return self.conn

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()

    @contextmanager
    def transaction(self):
        if self.readonly:
            raise sqlite3.OperationalError('登记表只读')
        with self.conn:
            yield self.conn

    def get_document(self,doc_id: str):
        if not ID.fullmatch(doc_id):
            raise ValueError('非法 doc_id')
        row = self.conn.execute('SELECT * FROM documents WHERE doc_id=?',(doc_id,)).fetchone()
        return dict(row) if row else None

    get = get_document

    def list_documents(self,*,status=None,limit=None):
        if status is not None and status not in STATUSES:
            raise ValueError('非法 status')
        if limit is not None and not 1<=limit<=1000:
            raise ValueError('非法 limit')
        sql = 'SELECT * FROM documents'+(' WHERE status=?' if status else '')+' ORDER BY source_path'
        args = (status,) if status else ()
        if limit is not None:
            sql, args = sql+' LIMIT ?', args+(limit,)
        return [dict(row) for row in self.conn.execute(sql,args)]

    def _write(self,record):
        names = list(record)
        sql = 'INSERT INTO documents ('+','.join(names)+') VALUES ('+','.join('?' for _ in names)+')'
        sql += ' ON CONFLICT(doc_id) DO UPDATE SET '+','.join(f'{n}=excluded.{n}' for n in names if n!='doc_id')
        self.conn.execute(sql,[record[n] for n in names])
        self.conn.execute('DELETE FROM doc_titles WHERE doc_id=?',(record['doc_id'],))
        self.conn.execute('INSERT INTO doc_titles(doc_id,title,filename) VALUES (?,?,?)',
            (record['doc_id'],record['title'],_filename(record['source_path'])))

    def register(self,source_path: str, content_hash: str, *, file_type=None,parser='text',title=None):
        """返回 (record, changed)。哈希未变不增版本；deleted 重现必须重走入库。"""
        doc_id = make_doc_id(source_path)
        previous = self.get_document(doc_id)
        if previous and previous['content_hash']==content_hash and previous['status']!='deleted':
            return previous,False
        changed_content = previous is not None and previous['content_hash']!=content_hash
        record = dict(doc_id=doc_id,source_path=source_path,title=title or _filename(source_path),
            file_type=file_type or PurePosixPath(source_path).suffix.lstrip('.').lower(),parser=parser,
            content_hash=content_hash,version=(previous['version']+int(changed_content) if previous else 1),
            status='pending',mineru_batch_id=None,pages=None,duplicate_of=None,error=None,
            updated_at=datetime.now(timezone.utc).isoformat())
        with self.transaction():
            self._write(record)
        return record,True

    def update(self,doc_id: str, **changes):
        """只允许白名单列名，所有值参数绑定；同时维护 FTS。"""
        if unknown := set(changes)-FIELDS:
            raise ValueError(f'非法更新字段：{sorted(unknown)}')
        if 'status' in changes and changes['status'] not in STATUSES:
            raise ValueError('非法 status')
        record = self.get_document(doc_id)
        if record is None:
            raise KeyError(doc_id)
        record.update(changes)
        record['updated_at'] = datetime.now(timezone.utc).isoformat()
        with self.transaction():
            self._write(record)
        return record

    def set_status(self,doc_id: str,status: str,**changes):
        return self.update(doc_id,status=status,**changes)

    def delete(self,doc_id: str, *, purge=False):
        """默认逻辑删除以通知向量层；purge 用于明确的永久清理。"""
        if not purge:
            return self.set_status(doc_id,'deleted')
        if not ID.fullmatch(doc_id):
            raise ValueError('非法 doc_id')
        with self.transaction():
            self.conn.execute('DELETE FROM doc_titles WHERE doc_id=?',(doc_id,))
            self.conn.execute('DELETE FROM documents WHERE doc_id=?',(doc_id,))

    def mark_missing(self,present_paths: set[str]):
        """调用方只在完整扫描后使用；limit批次不得将未扫描文件标删除。"""
        missing = [r['doc_id'] for r in self.list_documents() if r['source_path'] not in present_paths and r['status']!='deleted']
        for doc_id in missing:
            self.delete(doc_id)
        return missing

    def resumable(self):
        return [r for r in self.list_documents(status='parsing') if r['mineru_batch_id']]

    def exact_duplicate(self,content_hash: str,*,exclude_doc_id=None):
        row = self.conn.execute('''SELECT * FROM documents WHERE content_hash=?
            AND status IN ('cleaned','indexed') AND doc_id!=? ORDER BY source_path LIMIT 1''',
            (content_hash,exclude_doc_id or '')).fetchone()
        return dict(row) if row else None

    def title_candidates(self,query: str,*,limit=20):
        """仅 indexed；精确优先；不足3字符不模糊；FTS语法完全转义。"""
        if not 1<=limit<=1000:
            raise ValueError('非法 limit')
        phrase = normalize_title(query)
        if not phrase:
            return []
        rows = self.conn.execute('''SELECT d.*, t.filename FROM documents d
            JOIN doc_titles t ON d.doc_id=t.doc_id WHERE d.status='indexed' ORDER BY d.source_path''').fetchall()
        exact = [dict(r) for r in rows if phrase in (normalize_title(r['title']),normalize_title(r['filename']))]
        if exact:
            return exact[:limit]
        if len(phrase)<3:
            return []
        escaped = '"'+phrase.replace('"','""')+'"'
        return [dict(r) for r in self.conn.execute('''SELECT d.*,t.filename FROM doc_titles t
            JOIN documents d ON d.doc_id=t.doc_id WHERE doc_titles MATCH ?
            AND d.status='indexed' ORDER BY t.rank LIMIT ?''',(escaped,limit))]
