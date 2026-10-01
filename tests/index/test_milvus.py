from ragagent.index.milvus import INDEXED, ensure_collection, wait_indexed
import pytest


class IndexClient:
    def __init__(self, total=1, indexed=1, state='Finished'):
        self.total, self.indexed, self.state = total, indexed, state

    def describe_index(self, coll, field):
        return {'total_rows': self.total, 'indexed_rows': self.indexed, 'state': self.state}


def test_wait_requires_positive_fully_indexed_sealed_rows():
    assert set(wait_indexed(IndexClient(), 'l1_test')) == set(INDEXED)
    for client in (IndexClient(total=0, indexed=0), IndexClient(total=1, indexed=0)):
        with pytest.raises(TimeoutError, match='DISKANN'):
            wait_indexed(client, 'l1_test', timeout_s=0)
    with pytest.raises(RuntimeError, match='失败'):
        wait_indexed(IndexClient(total=1, indexed=0, state='Failed'), 'l1_test')


def test_existing_dimension_checked_without_release():
    class Client:
        def has_collection(self, coll):
            return True
        def describe_collection(self, coll):
            return {'fields': [{'name': 'dense', 'params': {'dim': 4}}]}
    client = Client()
    assert not ensure_collection(client, 'l1_test', 4)
    with pytest.raises(ValueError, match='维度'):
        ensure_collection(client, 'l1_test', 8)
