"""离线校准观测记录：仅输出建议及证据，绝不修改config或索引。"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from eval.metrics import percentile, recall_at_k

WEIGHTS = tuple(round(.5 + .05 * i, 2) for i in range(7))
CHUNK_TARGETS = (256, 400, 600)


def _score(x):
    if isinstance(x, bool) or not isinstance(x, (float,int)) or not math.isfinite(x) or not 0 <= x <= 1:
        raise ValueError('score must be finite and in [0,1]')
    return float(x)


def threshold_scan(samples, *, objective):
    """inclusive score>=threshold；扫描所有会改变混淆矩阵的断点。"""
    if objective not in {'youden_j','f1'} or not samples:
        raise ValueError('threshold scan requires samples and a known objective')
    pairs = [(_score(s['score']),s['label']) for s in samples]
    if any(type(label) is not bool for _,label in pairs):
        raise ValueError('calibration labels must be booleans')
    positives, negatives = sum(label for _,label in pairs), sum(not label for _,label in pairs)
    if not positives or not negatives:
        raise ValueError('both positive and negative labels are required')
    thresholds = sorted({0.,1.,*(score for score,_ in pairs)})
    curve=[]
    for threshold in thresholds:
        tp=sum(s>=threshold and l for s,l in pairs)
        fp=sum(s>=threshold and not l for s,l in pairs)
        fn=positives-tp
        tpr, fpr = tp/positives, fp/negatives
        precision=tp/(tp+fp) if tp+fp else 0.
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        curve.append(dict(threshold=threshold,tp=tp,fp=fp,fn=fn,tn=negatives-fp,
                          tpr=tpr,fpr=fpr,precision=precision,f1=f1,youden_j=tpr-fpr))
    best=max(curve,key=lambda r:(r[objective],r['precision'],-r['threshold']))
    return {'recommended':best['threshold'],'objective':objective,'best':best,'curve':curve,
            'samples':len(pairs),'positive':positives,'negative':negatives}


def rerank_loss(trials):
    groups=defaultdict(list)
    for trial in trials:
        t=_score(trial['threshold'])
        candidates=trial['candidates']
        before={c['chunk_id'] for c in candidates}
        kept={c['chunk_id'] for c in candidates if _score(c['score'])>=t}
        gold=[set(g) for g in trial['gold']]
        discarded=sum(bool(g & before) and not bool(g & kept) for g in gold)
        groups[t].append((discarded,len(gold),sum(not bool(g & before) for g in gold)))
    return [{'threshold':t,'discarded_gold_quotes':sum(x[0] for x in xs),
             'gold_quotes':sum(x[1] for x in xs),'missed_before_rerank':sum(x[2] for x in xs),
             'discard_rate':sum(x[0] for x in xs)/sum(x[1] for x in xs) if sum(x[1] for x in xs) else None}
            for t,xs in sorted(groups.items())]


def dense_weight_scan(trials):
    """输入必须是每个权重实际检索出的排名，不模拟Milvus的非线性归一化。"""
    groups=defaultdict(list)
    for t in trials:
        weight=round(float(t['dense_weight']),2)
        if weight not in WEIGHTS:
            raise ValueError('dense weight trials must cover 0.5..0.8 in 0.05 steps')
        groups[weight].append(t)
    curve=[]
    for weight,rows in sorted(groups.items()):
        recall=lambda rs: sum(recall_at_k(r['ranked'],[set(g) for g in r['gold']],30) for r in rs)/len(rs) if rs else None
        keyword=[r for r in rows if 'keyword' in r.get('tags',[])]
        curve.append({'dense':weight,'sparse':round(1-weight,2),'recall_at_30':recall(rows),
                      'keyword_recall_at_30':recall(keyword),'samples':len(rows)})
    missing=sorted(set(WEIGHTS)-set(groups))
    best=max(curve,key=lambda r:(r['recall_at_30'],r['keyword_recall_at_30'] or 0.,-r['dense'])) if curve else None
    return {'recommended':best if not missing else None,'missing_weights':missing,'curve':curve}


def retrieval_tradeoff(trials, *, keys=('candidates','search_list'), required=()):
    groups=defaultdict(list)
    for t in trials:
        latency=t['latency_s']
        if isinstance(latency,bool) or not isinstance(latency,(int,float)) or not math.isfinite(latency) or latency<0:
            raise ValueError('latency must be finite and nonnegative')
        groups[tuple(t[k] for k in keys)].append(t)
    rows=[]
    for key,ts in sorted(groups.items()):
        rows.append({**dict(zip(keys,key)), 'recall_at_8':sum(recall_at_k(t['ranked'],[set(g) for g in t['gold']],8) for t in ts)/len(ts),
                     'p95_latency_s':percentile([t['latency_s'] for t in ts],.95),'samples':len(ts)})
    frontier=[r for r in rows if not any(s['recall_at_8']>=r['recall_at_8'] and s['p95_latency_s']<=r['p95_latency_s']
              and (s['recall_at_8']>r['recall_at_8'] or s['p95_latency_s']<r['p95_latency_s']) for s in rows)]
    missing=[v for v in required if not any(r[keys[0]]==v for r in rows)]
    return {'curve':rows,'pareto_frontier':frontier,'missing_values':missing,
            'recommended':max(frontier,key=lambda r:(r['recall_at_8'],-r['p95_latency_s'])) if frontier and not missing else None}


def token_safety_factor(samples, *, minimum=1000):
    ratios,ids=[],set()
    for row in samples:
        cid=row['chunk_id']
        if cid in ids:
            raise ValueError('token safety samples must be distinct chunks')
        ids.add(cid)
        tokens,prompt,overhead=row['tokens'],row['prompt_tokens'],row['fixed_overhead']
        if any(type(v) is not int for v in (tokens,prompt,overhead)) or tokens<=0 or overhead<0 or prompt<=overhead:
            raise ValueError('invalid token usage or fixed overhead')
        ratios.append((prompt-overhead)/tokens)
    measured=percentile(ratios,.95) if ratios else None
    return {'recommended':measured if len(samples)>=minimum else None,'observed_p95':measured,
            'samples':len(samples),'required':minimum,'verified':len(samples)>=minimum}


def calibrate(data, *, mode):
    if mode not in {'real','synthetic'}:
        raise ValueError('mode must be real or synthetic')
    report={'mode':mode,'config_modified':False,'provenance':data.get('provenance'), 'failures':data.get('failures',[]), 'sections':{},'missing':[]}
    specs={'grade.pass_score':('grade_samples',lambda x:threshold_scan(x,objective='youden_j')),
           'postprocess.support_threshold':('support_pairs',lambda x:threshold_scan(x,objective='f1')),
           'rerank.threshold':('rerank_trials',rerank_loss),
           'retrieval.weights':('weight_trials',dense_weight_scan),
           'retrieval.candidates/search_list':('retrieval_trials',retrieval_tradeoff),
           'context.token_safety_factor':('token_samples',token_safety_factor),
           'chunking.target_tokens':('chunk_trials',lambda x:retrieval_tradeoff(x,keys=('target_tokens',),required=CHUNK_TARGETS))}
    for name,(key,func) in specs.items():
        if not data.get(key):
            report['missing'].append(name)
        else:
            try:
                report['sections'][name]=func(data[key])
            except ValueError as exc:
                # 某项单类/无效观测不能丢掉其他六项实测证据。
                report['sections'][name]={'verified':False,'recommended':None,
                                          'samples':len(data[key]),'error':str(exc)}
                report['missing'].append(name)
    report['real_calibration_complete']= mode=='real' and bool(report['provenance']) and not report['missing'] and not report['failures'] and all(
        not v.get('missing_weights') and not v.get('missing_values') and v.get('verified',True)
        for v in report['sections'].values() if isinstance(v,dict))
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='分布/排名/延迟/token实测记录JSON；见eval/README.md')
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['synthetic','real'],required=True)
    args=p.parse_args(argv)
    report=calibrate(json.loads(Path(args.input).read_text()),mode=args.mode)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(f"mode={args.mode} sections={len(report['sections'])} missing={len(report['missing'])} config_modified=False")
    return 0 if args.mode=='synthetic' or report['real_calibration_complete'] else 1


if __name__=='__main__':
    raise SystemExit(main())
