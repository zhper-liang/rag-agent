"""Visualized-BGE 编码服务的桩（总揽.md §4.4）。

文本 hash → 可复现的归一化向量：只保证确定性，不保证同义改写相近。
接口与真实服务一致（向量化建库.md §4.2），重排接口一并挂载。
"""

import hashlib
import math
import struct

from fastapi import FastAPI

from ragagent.index.models import EmbedImageRequest, EmbedResponse, EmbedTextRequest
from tests.stubs.rerank_server import router as rerank_router

MODEL = "stub-embed"


def vector(key: str, dim: int) -> list[float]:
    """sha256 计数器模式生成 dim 个 [-1, 1) 的数，再做 L2 归一化。"""
    raw = b"".join(hashlib.sha256(f"{i}:{key}".encode()).digest() for i in range(math.ceil(dim / 8)))
    xs = [u / 2**31 - 1 for u in struct.unpack(f"<{dim}I", raw[:dim * 4])]
    norm = math.sqrt(sum(x * x for x in xs))
    return [x / norm for x in xs]


def create_app(dim: int = 1024) -> FastAPI:
    app = FastAPI(title="embed-stub")

    @app.post("/embed/text")
    def embed_text(req: EmbedTextRequest) -> EmbedResponse:
        return EmbedResponse(vectors=[vector(t, dim) for t in req.texts])

    @app.post("/embed/image")
    def embed_image(req: EmbedImageRequest) -> EmbedResponse:
        # 不读图片文件：路径校验由 EmbedItem 完成，向量只由路径和文字决定
        return EmbedResponse(vectors=[vector(f"img:{i.image_path}\n{i.text}", dim) for i in req.items])

    @app.get("/health")
    def health() -> dict:
        return {"model": MODEL, "version": f"{MODEL}@{dim}", "device": "cpu", "dim": dim}

    app.include_router(rerank_router)
    return app


if __name__ == "__main__":
    from pathlib import Path

    import yaml

    from tests.stubs.serve import main
    cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "config.yaml").read_text(encoding="utf-8"))
    main(create_app(cfg["index"]["dim"]), cfg["services"]["embed"]["port"])
