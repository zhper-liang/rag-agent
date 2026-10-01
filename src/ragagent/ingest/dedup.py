"""文件SHA256、全文字符5-gram MinHash；签名带scheme，禁止pickle。"""
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from datasketch import MinHash

SCHEME = 'affine64'
NUM_PERM = 128


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def minhash(text: str) -> MinHash | None:
    value = re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).casefold()
    if not value:
        return None
    shingle = 5
    sketch = MinHash(num_perm=NUM_PERM, seed=1, scheme=SCHEME)
    pieces = {value[i:i+shingle] for i in range(max(1, len(value)-shingle+1))}
    sketch.update_batch([piece.encode('utf-8') for piece in sorted(pieces)])
    return sketch


class DocumentDedup:
    """内部表不改变documents/FTS5冻结契约；只匹配当前可检索版本。"""
    def __init__(self, registry, *, threshold=.9):
        self.registry, self.threshold = registry, threshold
        with registry.transaction():
            registry.conn.execute('''CREATE TABLE IF NOT EXISTS ingest_sketches(
                doc_id TEXT PRIMARY KEY,version INTEGER NOT NULL,scheme TEXT NOT NULL,
                hashvalues TEXT NOT NULL,FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE)''')

    def duplicate_of(self, text: str, *, exclude_doc_id=None) -> str | None:
        sketch = minhash(text)
        if sketch is None:
            return None
        rows = self.registry.conn.execute('''SELECT s.* FROM ingest_sketches s JOIN documents d USING(doc_id)
            WHERE d.status IN ('cleaned','indexed') AND s.version=d.version AND s.scheme=? AND s.doc_id!=?
            ORDER BY d.source_path''', (SCHEME, exclude_doc_id or ''))
        for row in rows:
            values = json.loads(row['hashvalues'])
            candidate = MinHash(num_perm=NUM_PERM, seed=1, scheme=SCHEME, hashvalues=values)
            if sketch.jaccard(candidate) >= self.threshold:
                return row['doc_id']
        return None

    def remember(self, doc_id: str, version: int, text: str):
        sketch = minhash(text)
        with self.registry.transaction():
            if sketch is None:
                self.registry.conn.execute('DELETE FROM ingest_sketches WHERE doc_id=?', (doc_id,))
                return
            self.registry.conn.execute('''INSERT INTO ingest_sketches VALUES (?,?,?,?)
                ON CONFLICT(doc_id) DO UPDATE SET version=excluded.version,scheme=excluded.scheme,hashvalues=excluded.hashvalues''',
                (doc_id, version, SCHEME, json.dumps([int(x) for x in sketch.hashvalues])))
