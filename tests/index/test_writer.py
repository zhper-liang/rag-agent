import pytest

from ragagent.index.writer import delete_document, index_document
from tests.index.helpers import Client, Encoder, chunks, config, document


def test_upsert_then_delete_old_versions_and_routes():
    client, emb, doc = Client(), Encoder(), document(2)
    parts = chunks(doc)
    cfg = config(**{'index.dim': 4, 'index.insert_batch': 1}).index
    assert index_document(client, 'l1_test', emb, doc, parts, cfg=cfg) == len(parts)
    assert [c[0] for c in client.calls] == ['upsert'] * len(parts) + ['delete']
    assert client.calls[-1][2] == 'doc_id == "0123456789abcdef" and version < 2'
    assert emb.calls[0][1] == [c.embed_text for c in parts if c.type != 'image']
    assert emb.calls[1][1] == [(c.image_path, c.embed_text) for c in parts if c.type == 'image']
    assert all('sparse' not in r for r in client.rows.values())
    assert all(r['page_start'] == r['page_end'] == -1 for r in client.rows.values())
    assert index_document(client, 'l1_test', emb, doc, parts, cfg=cfg) == len(parts)
    assert len(client.rows) == len(parts)


def test_partial_write_failure_keeps_old_version():
    client, emb, doc = Client(fail_batch=2), Encoder(), document(2)
    cfg = config(**{'index.dim': 4, 'index.insert_batch': 1}).index
    with pytest.raises(RuntimeError, match='partial'):
        index_document(client, 'l1_test', emb, doc, chunks(doc), cfg=cfg)
    assert client.rows  # 至少一批确实写了，随后失败
    assert not any(c[0] == 'delete' for c in client.calls)
    client.fail_batch = None
    assert index_document(client, 'l1_test', emb, doc, chunks(doc), cfg=cfg) == len(chunks(doc))


@pytest.mark.parametrize('mutation', [
    {'doc_id': 'deadbeefdeadbeef'}, {'version': 2}, {'seq': 10},
    {'chunk_id': 'wrong'}, {'text': '中' * 23000},
    {'image_ids': ['x'] * 33}, {'related_ids': ['x' * 65]},
])
def test_invalid_chunk_never_writes(mutation):
    client, emb, doc = Client(), Encoder(), document()
    parts = chunks(doc)
    parts[0] = parts[0].model_copy(update=mutation)
    with pytest.raises(ValueError):
        index_document(client, 'l1_test', emb, doc, parts)
    assert client.calls == [] and emb.calls == []


def test_duplicate_ids_empty_and_filter_injection():
    client, emb, doc = Client(), Encoder(), document()
    part = chunks(doc)[0]
    with pytest.raises(ValueError, match='重复'):
        index_document(client, 'l1_test', emb, doc, [part, part])
    assert index_document(client, 'l1_test', emb, doc, []) == 0
    bad = doc.model_copy(update={'doc_id': '" or true'})
    with pytest.raises(ValueError):
        index_document(client, 'l1_test', emb, bad, [])
    with pytest.raises(ValueError):
        delete_document(client, 'l1_test', '" or true')
    assert client.calls == []


def test_bad_encoder_result_never_writes():
    client, emb, doc = Client(), Encoder(), document()
    emb.text = lambda texts: []
    with pytest.raises(ValueError, match='数量'):
        index_document(client, 'l1_test', emb, doc, chunks(doc))
    assert client.calls == []


def test_underreported_upsert_never_deletes_old_versions():
    client, emb, doc = Client(), Encoder(), document(2)
    client.upsert = lambda *args: {'upsert_count': 0}
    with pytest.raises(RuntimeError, match='确认'):
        index_document(client, 'l1_test', emb, doc, chunks(doc))
    assert not any(c[0] == 'delete' for c in client.calls)
