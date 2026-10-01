"""§11 只标记重复，不删链；跨文档副本保留可检索。"""
import hashlib
import unicodedata
from collections import defaultdict


def content_hash(text):
    normalized = ''.join(unicodedata.normalize('NFKC',text).split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def mark_duplicates(chunks):
    seen = set()
    for c in chunks:
        if c.type == 'image':
            continue
        if c.content_hash in seen:
            c.searchable = False
            c.meta['duplicate'] = True
        seen.add(c.content_hash)
    return chunks


def mark_boilerplate(documents, threshold):
    by_hash = defaultdict(set)
    for chunks in documents:
        for c in chunks:
            if c.type != 'image':
                by_hash[c.content_hash].add(c.doc_id)
    for chunks in documents:
        for c in chunks:
            if c.type != 'image' and len(by_hash[c.content_hash]) > threshold:
                c.searchable = False
                c.meta['boilerplate'] = True
    return documents
