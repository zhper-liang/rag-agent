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


class AliasClient:
    def __init__(self, *, collections=(), error=None, aliases=None):
        self.collections, self.error = list(collections), error
        self.aliases = {'aliases': []} if aliases is None else aliases
        self.writes = []

    def list_collections(self):
        return self.collections

    def describe_alias(self, alias):
        if self.error:
            raise self.error
        return {'alias': alias, 'collection': 'l1_old'}

    def list_aliases(self):
        return self.aliases

    def create_alias(self, **kw):
        self.writes.append(('create', kw))

    def alter_alias(self, **kw):
        self.writes.append(('alter', kw))


def test_alias_conflict_preserves_physical_collection_without_writes():
    from ragagent.index.milvus import switch_alias
    client = AliasClient(collections=['rag_chunks'])
    with pytest.raises(ValueError, match='实体collection冲突'):
        switch_alias(client, 'l1_new')
    assert client.writes == []


@pytest.mark.parametrize('aliases', [{'aliases': [], 'db_name': 'default'}, []])
def test_missing_alias_creates_after_explicit_absence_check(aliases):
    from pymilvus import MilvusException
    from ragagent.index.milvus import switch_alias
    client = AliasClient(error=MilvusException(1600, 'alias not found'), aliases=aliases)
    switch_alias(client, 'l1_new')
    assert client.writes == [('create', {'collection_name': 'l1_new', 'alias': 'rag_chunks'})]


def test_existing_alias_alters_target():
    from ragagent.index.milvus import switch_alias
    client = AliasClient()
    switch_alias(client, 'l1_new')
    assert client.writes == [('alter', {'collection_name': 'l1_new', 'alias': 'rag_chunks'})]


@pytest.mark.parametrize('code,aliases', [(65535, {'aliases': []}), (1600, {'aliases': ['rag_chunks']})])
def test_alias_errors_or_inconsistent_inventory_never_create(code, aliases):
    from pymilvus import MilvusException
    from ragagent.index.milvus import switch_alias
    error = MilvusException(code, 'describe failed')
    client = AliasClient(error=error, aliases=aliases)
    with pytest.raises(MilvusException) as caught:
        switch_alias(client, 'l1_new')
    assert caught.value is error
    assert client.writes == []
