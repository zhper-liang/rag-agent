from pathlib import Path
import yaml

from ragagent.config import Config
from ragagent.ingest.models import Block, CleanDocument
from ragagent.ingest.chunking.pipeline import run_chunk


def config(**updates):
    raw = yaml.safe_load(Path('config.yaml').read_text(encoding='utf-8'))
    for dotted, value in updates.items():
        keys = dotted.split('.')
        target = raw
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return Config.model_validate(raw)


def document(version=1):
    return CleanDocument(doc_id='0123456789abcdef', version=version, source_path='manual.md',
                         file_type='md', parser='text', title='部署手册', content_hash='a' * 64,
                         blocks=[Block(type='paragraph', text='Milvus部署方法与错误码E00042。'),
                                 Block(type='image', text='磁盘索引结构图', image_path='images/disk.png')])


def chunks(doc=None):
    doc = doc or document()
    return run_chunk(doc, cfg=config().chunking, count=len)


class Client:
    def __init__(self, fail_batch=None):
        self.calls = []
        self.fail_batch = fail_batch
        self.rows = {}

    def upsert(self, coll, batch):
        self.calls.append(('upsert', coll, batch))
        if sum(c[0] == 'upsert' for c in self.calls) == self.fail_batch:
            raise RuntimeError('injected partial write')
        self.rows.update((r['chunk_id'], r) for r in batch)
        return {'upsert_count': len(batch)}

    def delete(self, coll, filter):
        self.calls.append(('delete', coll, filter))
        return {'delete_count': 0}

    def flush(self, coll):
        self.calls.append(('flush', coll))


class Encoder:
    dim = 4
    version = 'test@4'

    def __init__(self):
        self.calls = []

    def text(self, texts):
        self.calls.append(('text', texts))
        return [[1., 0., 0., 0.] for _ in texts]

    def image(self, items):
        self.calls.append(('image', items))
        return [[0., 1., 0., 0.] for _ in items]
