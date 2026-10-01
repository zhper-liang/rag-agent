"""§9.3 相邻余弦及节内 z-score；均匀/过短节不产生断点。"""
from math import sqrt
from statistics import fmean, pstdev


def cosine(a, b):
    if len(a) != len(b):
        raise ValueError('语义向量维度不同')
    denominator = sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    return sum(x*y for x,y in zip(a,b)) / denominator if denominator else 0.0


def boundaries(texts, cfg, encode):
    if not cfg.enabled or len(texts) < cfg.min_units:
        return {}
    if encode is None:
        raise ValueError('语义切分已启用但未提供文本编码函数')
    vectors = encode(texts)
    if len(vectors) != len(texts):
        raise ValueError('编码结果数量不匹配')
    similarities = [cosine(a,b) for a,b in zip(vectors,vectors[1:])]
    mean, std = fmean(similarities), pstdev(similarities)
    if std < cfg.min_std:
        return {}
    return {i+1: ('break' if (s-mean)/std <= cfg.break_z else
                 'strong' if (s-mean)/std >= cfg.strong_z else 'neutral')
            for i,s in enumerate(similarities)}
