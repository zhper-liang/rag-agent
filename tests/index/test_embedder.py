import httpx
import pytest

from ragagent.index.embedder import Embedder
from tests.stubs.embed_server import vector


def test_real_http_stub_protocol_and_lifetime(embed_url):
    with Embedder(embed_url, dim=1024) as emb:
        assert emb.version == 'stub-embed@1024'
        assert emb.health()['device'] == 'cpu'
        texts = ['部署Milvus', '错误码 E00042']
        assert emb.text(texts) == [vector(t, 1024) for t in texts]
        assert emb.image([('doc/v1/images/a.png', '图说明')]) == [
            vector('img:doc/v1/images/a.png\n图说明', 1024)]
        assert emb.text([]) == emb.image([]) == []
        for path in ('../secret.png', '/tmp/a.png', 'doc/../../a.png'):
            with pytest.raises(ValueError):
                emb.image([(path, '')])
    assert emb.client.is_closed


@pytest.mark.parametrize('vectors', [[], [[1., 0.]], [[0.] * 4], [[2., 0., 0., 0.]],
                                     [[float('inf'), 0., 0., 0.]]])
def test_bad_responses_rejected(vectors):
    # inf不能直接作为JSON编码；以合法JSON数值1e999触发非有限向量。
    def handler(request):
        if vectors and vectors[0][0] == float('inf'):
            return httpx.Response(200, content=b'{"vectors":[[1e999,0,0,0]]}')
        return httpx.Response(200, json={'vectors': vectors})
    with Embedder('http://127.0.0.1', dim=4, transport=httpx.MockTransport(handler)) as emb:
        with pytest.raises(ValueError):
            emb.text(['a'])


def test_health_dimension_and_http_error():
    with Embedder('http://127.0.0.1', dim=4, transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={'dim': 8, 'version': 'bad'}))) as emb:
        with pytest.raises(ValueError, match='维度'):
            emb.health()
    with Embedder('http://127.0.0.1', dim=4, transport=httpx.MockTransport(
            lambda _: httpx.Response(503))) as emb:
        with pytest.raises(httpx.HTTPStatusError):
            emb.text(['a'])
