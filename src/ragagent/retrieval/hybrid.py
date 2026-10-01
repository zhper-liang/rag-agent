"""批量编码、多查询并发 hybrid；逐查询失败时降级为 dense 一次。"""

from __future__ import annotations

import asyncio
import logging
import math
from time import perf_counter

from pymilvus import AnnSearchRequest, WeightedRanker

from ragagent.index.models import EmbedResponse
from ragagent.retrieval.filters import build_expr
from ragagent.retrieval.http import post_json
from ragagent.retrieval.merge import fetch_candidates, merge
from ragagent.retrieval.models import RetrievalResult

log = logging.getLogger(__name__)
LIGHT_FIELDS = ["chunk_id", "doc_id", "version", "type"]


class RetrievalUnavailable(RuntimeError):
    status_code = 503


def _hits(response):
    return [{**h["entity"], "score": float(h["distance"])} for h in response[0]]


def hybrid_one(client, coll: str, query: str, qvec: list[float], expr: str, cfg) -> list[dict]:
    dense = AnnSearchRequest(data=[qvec], anns_field="dense",
                             param={"metric_type": "COSINE", "params": {"search_list": cfg.search_list}},
                             limit=cfg.dense_top_k, expr=expr)
    sparse = AnnSearchRequest(data=[query], anns_field="sparse", param={"metric_type": "BM25"},
                              limit=cfg.sparse_top_k, expr=expr)
    return _hits(client.hybrid_search(
        collection_name=coll, reqs=[dense, sparse],
        ranker=WeightedRanker(cfg.weights.dense, cfg.weights.sparse),
        limit=cfg.per_query_limit, output_fields=LIGHT_FIELDS, timeout=cfg.timeout_ms / 1000,
    ))


def dense_one(client, coll: str, qvec: list[float], expr: str, cfg) -> list[dict]:
    return _hits(client.search(
        collection_name=coll, data=[qvec], anns_field="dense", filter=expr,
        search_params={"metric_type": "COSINE", "params": {"search_list": cfg.search_list}},
        limit=cfg.dense_top_k, output_fields=LIGHT_FIELDS, timeout=cfg.timeout_ms / 1000,
    ))


async def hybrid_many(client, coll: str, queries: list[str], vectors: list[list[float]], expr: str, cfg, *, metrics=None):
    if len(queries) != len(vectors):
        raise ValueError("查询向量数量不一致")

    async def one(query, vector):
        try:
            result = await asyncio.wait_for(asyncio.to_thread(hybrid_one, client, coll, query, vector, expr, cfg),
                                            cfg.timeout_ms / 1000)
            return result, False
        except Exception as exc:
            log.warning("hybrid failed_type=%s; retry dense once", type(exc).__name__)
            if metrics is not None:
                metrics.degraded.labels(kind="hybrid").inc()
            result = await asyncio.wait_for(asyncio.to_thread(dense_one, client, coll, vector, expr, cfg),
                                            cfg.timeout_ms / 1000)
            return result, True

    outcomes = await asyncio.gather(*(one(q, v) for q, v in zip(queries, vectors)), return_exceptions=True)
    succeeded = [o for o in outcomes if not isinstance(o, BaseException)]
    for outcome in outcomes:
        if isinstance(outcome, asyncio.CancelledError):
            raise outcome
        if isinstance(outcome, BaseException):
            log.warning("query failed_type=%s; ignored", type(outcome).__name__)
    if not succeeded:
        raise RetrievalUnavailable("所有检索查询均失败")
    return [r for r, _ in succeeded], any(dense_only for _, dense_only in succeeded)


async def retrieve(queries: list[str], filters: dict | None, deps):
    """独立于图的检索入口 -> (list[Hit], RetrievalResult)。"""
    cfg = deps.cfg
    expr = build_expr(filters)  # 必须早于任何服务调用，非法过滤不接触 Milvus。
    if not queries:
        return [], RetrievalResult()
    timings = {}
    start = perf_counter()
    try:
        payload = await post_json(deps.embed_client, cfg, "/embed/text", {"texts": queries},
                                  cfg.retrieval.timeout_ms / 1000)
        vectors = EmbedResponse.model_validate(payload).vectors
        if (len(vectors) != len(queries) or any(len(v) != cfg.index.dim for v in vectors)
                or any(not math.isfinite(x) for v in vectors for x in v)):
            raise ValueError("编码向量形状或数值非法")
    except Exception as exc:
        raise RetrievalUnavailable("编码服务不可用") from exc
    timings["embed"] = round((perf_counter() - start) * 1000)
    start = perf_counter()
    if deps.milvus is None:
        raise RetrievalUnavailable("Milvus 客户端未配置")
    results, dense_only = await hybrid_many(deps.milvus, cfg.services.milvus.collection, queries, vectors,
                                           expr, cfg.retrieval, metrics=deps.metrics)
    timings["hybrid"] = round((perf_counter() - start) * 1000)
    selected = merge(results, cfg.retrieval.candidates)
    start = perf_counter()
    try:
        candidates = await asyncio.to_thread(fetch_candidates, deps.milvus, cfg.services.milvus.collection,
                                              selected, cfg.retrieval)
    except Exception as exc:
        raise RetrievalUnavailable("检索候选回表失败") from exc
    timings["fetch"] = round((perf_counter() - start) * 1000)
    return candidates, RetrievalResult(dense_only=dense_only, timings_ms=timings)
