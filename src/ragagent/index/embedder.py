"""编码服务的同步客户端；离线切分与写入使用同一份文本/图文协议。"""
from __future__ import annotations

from math import isfinite, sqrt

import httpx

from ragagent.config import Config, load_config
from ragagent.index.models import EmbedImageRequest, EmbedItem, EmbedResponse, EmbedTextRequest


class Embedder:
    def __init__(self, base_url: str | None = None, *, cfg: Config | None = None,
                 dim: int | None = None, timeout: float = 120, transport=None):
        if base_url is None or dim is None:
            cfg = cfg or load_config()
        self.dim = dim if dim is not None else cfg.index.dim
        if self.dim <= 0:
            raise ValueError('向量维度必须为正')
        if base_url is None:
            service = cfg.services.embed
            base_url = f'http://{service.host}:{service.port}'
        self.client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport,
                                   trust_env=False)
        self._version: str | None = None

    def health(self) -> dict:
        response = self.client.get('/health')
        response.raise_for_status()
        health = response.json()
        if health.get('dim', self.dim) != self.dim:
            raise ValueError('编码服务维度与index.dim不一致')
        version = health.get('version')
        if not isinstance(version, str) or not version:
            raise ValueError('编码服务未提供模型版本')
        self._version = version
        return health

    @property
    def version(self) -> str:
        if self._version is None:
            self.health()
        return self._version

    def _embed(self, route: str, payload: dict, count: int) -> list[list[float]]:
        if not count:
            return []
        response = self.client.post(route, json=payload)
        response.raise_for_status()
        vectors = EmbedResponse.model_validate(response.json()).vectors
        if len(vectors) != count:
            raise ValueError('编码结果数量不匹配')
        for vector in vectors:
            if len(vector) != self.dim or not all(isfinite(x) for x in vector):
                raise ValueError('编码向量维度错误或含非有限值')
            norm = sqrt(sum(x * x for x in vector))
            if abs(norm - 1) > 0.01:
                raise ValueError('编码服务必须返回L2归一化向量')
        return vectors

    def text(self, texts: list[str]) -> list[list[float]]:
        request = EmbedTextRequest(texts=texts)
        return self._embed('/embed/text', request.model_dump(), len(texts))

    def image(self, items: list[tuple[str, str]]) -> list[list[float]]:
        request = EmbedImageRequest(items=[EmbedItem(image_path=p, text=t) for p, t in items])
        return self._embed('/embed/image', request.model_dump(), len(items))

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
