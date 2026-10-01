"""批量重排、阈值/文档上限/top_n 筛选，失败保留融合 top_n。"""

import logging
import math
from time import perf_counter

from ragagent.index.models import RerankResponse
from ragagent.retrieval.http import post_json
from ragagent.retrieval.models import Hit, RetrievalResult

log = logging.getLogger(__name__)


def select(hits: list[dict], scores: list[float], cfg) -> list[dict]:
    if len(hits) != len(scores) or any(not math.isfinite(s) or s < 0 or s > 1 for s in scores):
        raise ValueError("重排分数数量/数值非法")
    ranked = sorted(({**h, "rerank_score": float(s)} for h, s in zip(hits, scores)),
                    key=lambda h: (-h["rerank_score"], h["chunk_id"]))
    out, per_doc = [], {}
    for h in ranked:
        if h["rerank_score"] < cfg.threshold:
            break
        doc = h["doc_id"]
        if per_doc.get(doc, 0) >= cfg.per_doc_cap:
            continue
        per_doc[doc] = per_doc.get(doc, 0) + 1
        out.append(h)
        if len(out) >= cfg.top_n:
            break
    return out


async def rerank(question: str, candidates, deps, retrieval: dict | RetrievalResult | None = None):
    cfg = deps.cfg
    hits = [Hit.model_validate(h).model_dump() for h in candidates]
    result = RetrievalResult.model_validate(retrieval or {}).model_copy(deep=True)
    result.degraded, result.top_score = False, 0.0
    start = perf_counter()
    if hits:
        try:
            payload = await post_json(deps.rerank_client, cfg, "/rerank",
                                      {"query": question, "passages": [h["embed_text"] for h in hits],
                                       "max_length": cfg.rerank.max_length}, cfg.rerank.timeout_ms / 1000)
            scores = RerankResponse.model_validate(payload).scores
            selected = select(hits, scores, cfg.rerank)
            result.top_score = max(scores, default=0.0)  # 筛选以前的最高分。
        except Exception as exc:
            log.warning("rerank failed_type=%s; use fusion", type(exc).__name__)
            result.degraded = True
            if deps.metrics is not None:
                deps.metrics.degraded.labels(kind="rerank").inc()
            selected = [{**h, "rerank_score": None} for h in sorted(hits, key=lambda h: (-h["score"], h["chunk_id"]))[:cfg.rerank.top_n]]
    else:
        selected = []
    result.timings_ms["rerank"] = round((perf_counter() - start) * 1000)
    return [Hit.model_validate(h) for h in selected], result
