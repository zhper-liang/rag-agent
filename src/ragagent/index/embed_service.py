"""内部串行编码/重排服务，协议与 P0 桩一致（向量化建库 §4.2）。"""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from math import isfinite, sqrt
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException

from ragagent.config import Config, load_config
from ragagent.index.models import (
    EmbedImageRequest, EmbedResponse, EmbedTextRequest,
    RerankPairsRequest, RerankRequest, RerankResponse,
)


def resolve_image(root: Path, relative: str) -> Path:
    """在读取图片前再次校验真实路径，挡住指向 clean 外的符号链接。"""
    path = Path(relative)
    if not relative or path.is_absolute() or '..' in path.parts or '\\' in relative:
        raise ValueError('图片必须使用clean目录内的相对路径')
    # Chunk.image_path 使用文档约定 clean/{doc_id}/v{version}/...，
    # HTTP 也接受直接相对 paths.clean 的路径；两者都限制在同一根目录内。
    if path.parts and path.parts[0] == 'clean':
        path = Path(*path.parts[1:])
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError('图片不在clean目录内或不是普通文件')
    return resolved


def _vectors(vectors: list[list[float]], size: int, dim: int) -> EmbedResponse:
    if len(vectors) != size:
        raise RuntimeError('模型返回的向量数量不匹配')
    for vector in vectors:
        if len(vector) != dim or not all(isfinite(x) for x in vector):
            raise RuntimeError('模型返回的向量维度错误或包含非有限值')
        if abs(sqrt(sum(x * x for x in vector)) - 1) > .01:
            raise RuntimeError('模型返回的向量未做L2归一化')
    return EmbedResponse(vectors=vectors)


def create_app(cfg: Config | None = None, *, backend=None, backend_factory=None) -> FastAPI:
    """backend 注入仅供本线测试；正常启动在 lifespan 中加载真实模型。"""
    cfg = cfg or load_config()
    root = Path(cfg.paths.clean).resolve()
    mutex = Lock()

    @asynccontextmanager
    async def lifespan(app):
        nonlocal backend
        owned = backend is None
        if owned:
            if backend_factory is None:
                from ragagent.index.gpu_models import GPUModels
                backend = GPUModels(cfg)
            else:
                backend = backend_factory(cfg)
        try:
            if backend.dim != cfg.index.dim:
                raise ValueError('模型维度与index.dim不一致')
            yield
        finally:
            if owned:
                backend.close()
                backend = None

    app = FastAPI(title='Visualized-BGE embedding and reranking', lifespan=lifespan)

    def require_backend():
        if backend is None:
            raise HTTPException(503, '编码模型尚未就绪')
        return backend

    @app.get('/health')
    def health() -> dict:
        with mutex:
            return require_backend().health()

    @app.post('/embed/text')
    def embed_text(req: EmbedTextRequest) -> EmbedResponse:
        with mutex:
            model = require_backend()
            return _vectors(model.text(req.texts) if req.texts else [], len(req.texts), model.dim)

    @app.post('/embed/image')
    def embed_image(req: EmbedImageRequest) -> EmbedResponse:
        try:
            items = [(str(resolve_image(root, item.image_path)), item.text) for item in req.items]
        except (ValueError, OSError):
            raise HTTPException(422, '图片必须是clean目录内存在的普通文件') from None
        with mutex:
            model = require_backend()
            try:
                vectors = model.image(items) if items else []
            except (ValueError, OSError):
                raise HTTPException(422, '无法解码图片') from None
            return _vectors(vectors, len(items), model.dim)

    def rerank_pairs(pairs, max_length) -> RerankResponse:
        with mutex:
            model = require_backend()
            if not 4 <= max_length <= model.max_length:
                raise HTTPException(422, 'max_length超出模型支持范围')
            scores = model.rerank(pairs, max_length) if pairs else []
            if len(scores) != len(pairs) or not all(isfinite(s) and 0 <= s <= 1 for s in scores):
                raise RuntimeError('模型返回的重排分数无效')
            return RerankResponse(scores=scores)

    @app.post('/rerank')
    def rerank(req: RerankRequest) -> RerankResponse:
        return rerank_pairs([(req.query, p) for p in req.passages], req.max_length)

    @app.post('/rerank/pairs')
    def pairs(req: RerankPairsRequest) -> RerankResponse:
        return rerank_pairs(req.pairs, req.max_length)

    return app


def main():
    parser = argparse.ArgumentParser(description='本地Visualized-BGE编码和bge-reranker服务')
    parser.add_argument('--config', default='config.yaml', help='项目配置文件')
    parser.add_argument('--visual-code', type=Path, default=Path('models/FlagEmbedding/research/visual_bge'),
                        help='官方FlagEmbedding仓库research/visual_bge目录')
    parser.add_argument('--text-model', type=Path, default=Path('models/BAAI/bge-m3'),
                        help='本地bge-m3配置和tokenizer目录')
    parser.add_argument('--rerank-model', type=Path, default=Path('models/BAAI/bge-reranker-v2-m3'),
                        help='本地bge-reranker-v2-m3权重和tokenizer目录')
    args = parser.parse_args()
    cfg = load_config(args.config)
    from ragagent.index.gpu_models import GPUModels
    import uvicorn
    app = create_app(cfg, backend_factory=lambda cfg: GPUModels(
        cfg, visual_code=args.visual_code, text_model=args.text_model, rerank_model=args.rerank_model))
    uvicorn.run(app, host=cfg.services.embed.host, port=cfg.services.embed.port, workers=1)


if __name__ == '__main__':
    main()
