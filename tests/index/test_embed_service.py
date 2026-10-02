from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
import time

from fastapi.testclient import TestClient
import pytest

from ragagent.index.embed_service import create_app
from tests.index.helpers import config


class Model:
    dim = 4
    max_length = 8192

    def __init__(self):
        self.calls = []
        self.active = 0
        self.peak_active = 0
        self.guard = Lock()
        self.closed = False

    def health(self):
        return {'model': 'test', 'version': 'test@1', 'device': 'cpu', 'dim': self.dim}

    def text(self, texts):
        with self.guard:
            self.active += 1
            self.peak_active = max(self.active, self.peak_active)
        time.sleep(.01)
        with self.guard:
            self.active -= 1
        self.calls.append(('text', texts))
        return [[1., 0., 0., 0.] for _ in texts]

    def image(self, items):
        self.calls.append(('image', items))
        return [[0., 1., 0., 0.] for _ in items]

    def rerank(self, pairs, max_length):
        self.calls.append(('rerank', pairs, max_length))
        return [.9 if q in p else .1 for q, p in pairs]

    def close(self):
        self.closed = True


@pytest.fixture
def service(tmp_path):
    root = tmp_path / 'clean'
    root.mkdir()
    model = Model()
    cfg = config(**{'paths.clean': str(root), 'index.dim': 4})
    with TestClient(create_app(cfg, backend=model)) as client:
        yield client, model, root


def test_protocol_and_order(service):
    client, model, root = service
    (root / 'test.png').write_bytes(b'fake-model-does-not-decode')
    assert client.get('/health').json()['version'] == 'test@1'
    assert client.post('/embed/text', json={'texts': ['a', 'b']}).json()['vectors'] == [[1., 0., 0., 0.]] * 2
    response = client.post('/embed/image', json={'items': [{'image_path': 'test.png', 'text': 'caption'}]})
    assert response.status_code == 200
    assert model.calls[-1] == ('image', [(str(root / 'test.png'), 'caption')])
    assert client.post('/rerank', json={'query': 'x', 'passages': ['y', 'x'], 'max_length': 512}).json() == {'scores': [.1, .9]}
    assert client.post('/rerank/pairs', json={'pairs': [['x', 'x'], ['x', 'z']], 'max_length': 512}).json() == {'scores': [.9, .1]}


@pytest.mark.parametrize('name', ['../outside.png', '/etc/passwd', 'missing.png', '..\\outside.png', ''])
def test_image_rejected_before_inference(service, name):
    client, model, _ = service
    response = client.post('/embed/image', json={'items': [{'image_path': name, 'text': 'caption'}]})
    assert response.status_code == 422
    assert model.calls == []


def test_symlink_escape_and_batch_validation(service, tmp_path):
    client, model, root = service
    outside = tmp_path / 'outside.png'
    outside.write_bytes(b'outside')
    (root / 'link.png').symlink_to(outside)
    (root / 'valid.png').write_bytes(b'valid')
    for name in ('link.png', '.'):
        response = client.post('/embed/image', json={'items': [
            {'image_path': 'valid.png', 'text': 'ok'}, {'image_path': name, 'text': 'bad'}]})
        assert response.status_code == 422
    assert model.calls == []


@pytest.mark.parametrize('length', [-1, 0, 3, 8193])
def test_invalid_rerank_budget(service, length):
    client, model, _ = service
    assert client.post('/rerank', json={'query': 'x', 'passages': ['x'], 'max_length': length}).status_code == 422
    assert model.calls == []


def test_empty_requests_do_not_infer(service):
    client, model, _ = service
    for route, data, key in [('/embed/text', {'texts': []}, 'vectors'),
                             ('/embed/image', {'items': []}, 'vectors'),
                             ('/rerank/pairs', {'pairs': [], 'max_length': 512}, 'scores')]:
        assert client.post(route, json=data).json() == {key: []}
    assert model.calls == []


def test_inference_is_serial_even_for_concurrent_http(service):
    client, model, _ = service
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda i: client.post('/embed/text', json={'texts': [str(i)]}), range(12)))
    assert all(r.status_code == 200 for r in responses)
    assert model.peak_active == 1
    assert len(model.calls) == 12


def test_model_lifecycle(tmp_path):
    model = Model()
    cfg = config(**{'paths.clean': str(tmp_path), 'index.dim': 4})
    with TestClient(create_app(cfg, backend_factory=lambda cfg: model)) as client:
        assert client.get('/health').status_code == 200
        assert not model.closed
    assert model.closed


def test_bad_model_output_is_rejected(service):
    client, model, _ = service
    model.text = lambda texts: [[float('nan'), 0., 0., 0.]]
    with pytest.raises(RuntimeError, match='非有限值'):
        client.post('/embed/text', json={'texts': ['x']})
    model.rerank = lambda pairs, length: [2.]
    with pytest.raises(RuntimeError, match='重排分数无效'):
        client.post('/rerank', json={'query': 'x', 'passages': ['x'], 'max_length': 512})


def test_chunk_image_path_with_clean_prefix(service):
    client, model, root = service
    relative = Path('0123456789abcdef/v1/images/figure.png')
    (root / relative).parent.mkdir(parents=True)
    (root / relative).write_bytes(b'fixture')
    response = client.post('/embed/image', json={'items': [
        {'image_path': 'clean/' + relative.as_posix(), 'text': 'caption'}]})
    assert response.status_code == 200
    assert model.calls[-1] == ('image', [(str(root / relative), 'caption')])
    assert client.post('/embed/image', json={'items': [
        {'image_path': 'clean/../outside.png', 'text': 'bad'}]}).status_code == 422
