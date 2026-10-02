"""离线验证显存不足时的批次恢复；不加载或伪称真实模型。"""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from ragagent.index.gpu_models import GPUModels


class OOM(RuntimeError):
    pass


class Inputs(dict):
    def to(self, _device):
        return self


def backend(capacity):
    model = GPUModels.__new__(GPUModels)
    model.device = 'cuda'
    model.batch_size = 8
    model.text_max_length = 8192
    calls, cleared = [], []
    model.torch = SimpleNamespace(inference_mode=nullcontext, cuda=SimpleNamespace(
        OutOfMemoryError=OOM, empty_cache=lambda: cleared.append(True)))
    model.tokenizer = lambda texts, **kwargs: Inputs(texts=texts)

    def encode(inputs):
        texts = inputs['texts']
        calls.append(list(texts))
        if len(texts) > capacity:
            raise OOM('simulated capacity')
        return [[float(len(t)), float(ord(t[0]))] for t in texts]

    model.visual = SimpleNamespace(encode_text=encode)
    model._rows = lambda values: values
    return model, calls, cleared


def test_oom_split_preserves_original_order_and_every_item():
    model, calls, cleared = backend(2)
    texts = ['zzzzz', 'a', 'bbb', 'cc', 'dddd']
    assert model.text(texts) == [[float(len(t)), float(ord(t[0]))] for t in texts]
    assert cleared
    successful = [t for batch in calls if len(batch) <= 2 for t in batch]
    assert sorted(successful) == sorted(texts)


def test_single_item_oom_is_not_hidden_or_retried_forever():
    model, calls, cleared = backend(0)
    with pytest.raises(OOM):
        model.text(['cannot-fit'])
    assert calls == [['cannot-fit']]
    assert not cleared


def test_empty_text_does_not_touch_gpu():
    model, calls, cleared = backend(0)
    assert model.text([]) == []
    assert not calls and not cleared
