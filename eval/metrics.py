"""检索指标：gold 是片段集合的列表，一个片段可能有多个替代 chunk。"""
import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], gold: Sequence[set[str]], k: int) -> float:
    """分母为 gold 片段数，而非其覆盖的 chunk 数；无 gold 不参与调用。"""
    if not gold:
        raise ValueError("recall requires nonempty gold")
    found = set(ranked[:k])
    return sum(bool(ids & found) for ids in gold) / len(gold)


def reciprocal_rank(ranked: Sequence[str], gold: Sequence[set[str]]) -> float:
    relevant = set().union(*gold) if gold else set()
    return next((1 / i for i, cid in enumerate(ranked, 1) if cid in relevant), 0.0)


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[set[str]], k: int) -> float:
    """二元片段相关度；一个片段的替代 chunk 不重复获益。

    同一排名命中多个独立 gold 时每片段各贡献一份 gain。理想排名通过
    相同候选覆盖关系贪心选取最大新增片段数，避免跨段替代 ID 重复获益。
    每个 gold 片段单独出现时退化为常规 binary nDCG。
    """
    if not gold or k <= 0:
        return 0.0

    def dcg(ids):
        covered, total = set(), 0.0
        for rank, cid in enumerate(ids[:k], 1):
            matched = {i for i, targets in enumerate(gold) if cid in targets} - covered
            total += len(matched) / math.log2(rank + 1)
            covered.update(matched)
        return total

    remaining = set().union(*gold)
    ideal, covered = [], set()
    while remaining and len(ideal) < k:
        cid = max(sorted(remaining), key=lambda c: sum(i not in covered and c in g for i, g in enumerate(gold)))
        covered.update(i for i, g in enumerate(gold) if cid in g)
        ideal.append(cid)
        remaining.remove(cid)
    denom = dcg(ideal)
    return min(1.0, dcg(list(ranked)) / denom) if denom else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """线性插值 percentile（空序列没有观测值，不能伪造 0）。"""
    if not values:
        raise ValueError("no observations")
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (pos - lo)
