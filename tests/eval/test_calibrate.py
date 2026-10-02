import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval.calibrate import (WEIGHTS, calibrate, dense_weight_scan, rerank_loss,
                           retrieval_tradeoff, threshold_scan, token_safety_factor)


def synthetic():
    pairs=[{'score':.1,'label':False},{'score':.4,'label':False},{'score':.5,'label':True},{'score':.9,'label':True}]
    rank={'ranked':['a'],'gold':[['a','a2']],'tags':['keyword'],'latency_s':.1}
    return {'grade_samples':pairs,'support_pairs':pairs,
            'rerank_trials':[{'threshold':.5,'candidates':[{'chunk_id':'a','score':.4},{'chunk_id':'a2','score':.6}], 'gold':[['a','a2'],['b']]}],
            'weight_trials':[{**rank,'dense_weight':w} for w in WEIGHTS],
            'retrieval_trials':[{**rank,'candidates':30,'search_list':100}],
            'chunk_trials':[{**rank,'target_tokens':v} for v in (256,400,600)],
            'token_samples':[{'chunk_id':f'c{i}','tokens':100,'prompt_tokens':130,'fixed_overhead':20} for i in range(1000)]}


def test_youden_j_and_f1_hand_calculated():
    for objective in ['youden_j','f1']:
        result=threshold_scan(synthetic()['grade_samples'],objective=objective)
        assert result['recommended']==.5
        assert result['best']['tp']==2 and result['best']['tn']==2
        assert result['best'][objective]==1
    with pytest.raises(ValueError): threshold_scan([{'score':.5,'label':True}],objective='f1')
    with pytest.raises(ValueError): threshold_scan([{'score':float('nan'),'label':False}],objective='f1')


def test_rerank_loss_fragment_not_chunk_denominator():
    # 2片段：{a,a2}留下a2不算丢；b原先没召回，不能归罪阈值。
    trials=synthetic()['rerank_trials']
    assert rerank_loss(trials)==[{'threshold':.5,'discarded_gold_quotes':0,'gold_quotes':2,'missed_before_rerank':1,'discard_rate':0}]
    trials[0]['threshold']=.7
    assert rerank_loss(trials)[0]['discard_rate']==.5


def test_weight_and_latency_scans_do_not_fake_missing_experiments():
    trials=synthetic()['weight_trials']
    assert dense_weight_scan(trials)['recommended']['dense']==.5
    assert dense_weight_scan(trials[:1])['recommended'] is None
    base=synthetic()['retrieval_trials'][0]
    rows=[base,{**base,'candidates':50,'latency_s':.2},{**base,'candidates':10,'latency_s':.05,'ranked':['x']}]
    result=retrieval_tradeoff(rows)
    assert len(result['pareto_frontier'])==2
    assert result['recommended']['candidates']==30


def test_token_p95_and_distinct_minimum():
    samples=[{'chunk_id':f'c{i}','tokens':10,'prompt_tokens':v+20,'fixed_overhead':20} for i,v in enumerate([10,20,30])]
    result=token_safety_factor(samples,minimum=3)
    assert result['recommended']==pytest.approx(2.9)
    assert token_safety_factor(samples)['recommended'] is None
    with pytest.raises(ValueError): token_safety_factor([samples[0],samples[0]])


def test_cli_all_seven_sections_never_writes_configuration(tmp_path):
    cfg_before=Path('config.yaml').read_bytes()
    data=synthetic(); src=tmp_path/'input.json'; out=tmp_path/'report.json'
    src.write_text(json.dumps(data))
    p=subprocess.run([sys.executable,'-m','eval.calibrate','--input',str(src),'--output',str(out),'--mode','synthetic'],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    report=json.loads(out.read_text())
    assert len(report['sections'])==7 and not report['missing']
    assert not report['real_calibration_complete'] and not report['config_modified']
    assert report['sections']['context.token_safety_factor']['recommended']==1.1
    assert Path('config.yaml').read_bytes()==cfg_before
    assert not calibrate(data,mode='real')['real_calibration_complete']  # 缺实测来源


def test_real_calibration_preserves_other_sections_when_labels_single_class():
    data = {'provenance': {'mode':'real'},
            'grade_samples':[{'score':.8,'label':True}],
            'support_pairs':[{'score':.7,'label':False}],
            'rerank_trials':[{'threshold':.3,'candidates':[{'chunk_id':'x','score':.8}], 'gold':[['x']]}],
            'failures':[{'phase':'support','error_type':'TimeoutError'}]}
    report = calibrate(data, mode='real')
    assert not report['real_calibration_complete']
    assert report['sections']['grade.pass_score']['verified'] is False
    assert 'postprocess.support_threshold' in report['missing']
    assert report['sections']['rerank.threshold']
    assert report['failures'] == data['failures']
