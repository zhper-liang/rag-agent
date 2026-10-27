"""引用格式离线回放不引入模型请求，也不冒充支持度指标。"""
from eval.citation_format_regression import replay


def test_replay_compatible_and_legacy_formats():
    rows = [dict(run='first', case=dict(id='q1', expect=dict(answerable=True)),
                 state=dict(answer='😀事实【1】。`arr[1]【2】`', context_blocks=[dict(cite=1)])),
            dict(run='first', case=dict(id='q2', expect=dict(answerable=True)),
                 state=dict(answer='事实[1]【99】。', context_blocks=[dict(cite=1)]))]
    report = replay(rows, heading='补充说明')
    assert report['support_revalidated'] is False
    assert report['runs']['first'] == dict(answers=2, answerable=2, legacy_citations=1,
                                          compatible_citations=3, recovered_zero_citation_answers=1,
                                          added_citations=2, added_unverified=1, added_invalid_number=1)
    assert len(report['changed']) == 2
    assert 'citation_support_rate' not in report


def test_replay_supplement_and_no_citations():
    rows = [dict(run='second', case=dict(id='q3', expect=dict(answerable=False)),
                 state=dict(answer='没有资料。\n补充说明\n建议【1】。', context_blocks=[]))]
    report = replay(rows, heading='补充说明')
    assert report['runs']['second']['added_in_supplement'] == 1
    assert report['runs']['second']['answerable'] == 0
    assert replay([], heading='补充说明')['changed'] == []
