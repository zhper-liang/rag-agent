"""RAGAS/出题共享本地HTTP编码适配；绝不回退远程默认embedding。"""
from __future__ import annotations

import math

import httpx
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    def __init__(self, base_url: str, *, dimension: int, timeout: float):
        self.base_url, self.dimension, self.timeout = base_url.rstrip('/'), dimension, timeout

    @classmethod
    def from_config(cls, cfg):
        svc = cfg.services.embed
        return cls(f'http://{svc.host}:{svc.port}', dimension=cfg.index.dim,
                   timeout=cfg.llm.scenes.eval.timeout_s)

    def _vectors(self, response, count):
        response.raise_for_status()
        vectors = response.json()['vectors']
        if len(vectors) != count or any(len(v) != self.dimension or any(not math.isfinite(x) for x in v) for v in vectors):
            raise ValueError('local embedding count/dimension/finite validation failed')
        return vectors

    def embed_documents(self, texts):
        if not texts:
            return []
        with httpx.Client(timeout=self.timeout) as client:
            return self._vectors(client.post(self.base_url + '/embed/text', json={'texts': texts}), len(texts))

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts):
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return self._vectors(await client.post(self.base_url + '/embed/text', json={'texts': texts}), len(texts))

    async def aembed_query(self, text):
        return (await self.aembed_documents([text]))[0]
