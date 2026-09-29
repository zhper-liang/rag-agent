"""bge-reranker-v2-m3 的桩：按词重叠度打分（开发计划.md §3.4）。

真实服务里重排和编码在同一个进程（向量化建库.md §4.2），所以这里提供 router，
由 embed_server 挂载；也可以单独启动。
"""

import re

from fastapi import APIRouter, FastAPI

from ragagent.index.models import RerankPairsRequest, RerankRequest, RerankResponse

# ASCII 单词整体算一个词，其余非空白字符（中文）逐字算
_TOKEN = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")

router = APIRouter()


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def score(query: str, passage: str, max_length: int) -> float:
    """query 的词有多少出现在 passage 里，0~1；和真实模型一样先按 max_length 截断。"""
    q = _tokens(query)[:max_length]
    p = set(_tokens(passage)[:max(max_length - len(q), 0)])
    return sum(t in p for t in q) / len(q) if q else 0.0


@router.post("/rerank")
def rerank(req: RerankRequest) -> RerankResponse:
    return RerankResponse(scores=[score(req.query, p, req.max_length) for p in req.passages])


@router.post("/rerank/pairs")
def rerank_pairs(req: RerankPairsRequest) -> RerankResponse:
    return RerankResponse(scores=[score(q, p, req.max_length) for q, p in req.pairs])


def create_app() -> FastAPI:
    app = FastAPI(title="rerank-stub")
    app.include_router(router)
    return app


if __name__ == "__main__":
    from tests.stubs.serve import main
    main(create_app(), 8002)
