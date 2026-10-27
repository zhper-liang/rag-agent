import asyncio
import copy

import pytest

from ragagent.config import load_config
from ragagent.context.models import ContextBlock
from ragagent.postprocess.final import build_final
from ragagent.postprocess.offsets import utf16_offset
from ragagent.postprocess.parse import check_format, find_supplement, mask_code, split_sentences
from ragagent.postprocess.plain import make_plain
from ragagent.postprocess.verify import VerifyDegraded, check_support, support_warning, token_windows


@pytest.fixture
def cfg():
    return load_config(env={k: 'test' for k in ['LLM_MAIN_KEY', 'LLM_MAIN_BACKUP_KEY', 'LLM_SMALL_KEY', 'LLM_SMALL_BACKUP_KEY']})


def block(cite=1, **kw):
    return ContextBlock(cite=cite, doc_id='0123456789abcdef', title='手册', section_path='配置',
                        page_start=2, page_end=3, text=kw.pop('text', '服务默认使用端口19530。'),
                        chunk_ids=['c1', 'c2'], hit_ids=['c1'], score=.8, **kw)


def parse(answer, cfg, blocks=None):
    masked = mask_code(answer)
    sents = split_sentences(answer, masked, find_supplement(masked, cfg.prompt.supplement_heading))
    check_format(sents, {1: block()} if blocks is None else blocks)
    return sents


@pytest.mark.parametrize('code', ['`arr[1]`', '`` x`arr[1]` ``', '```py\narr[1]\n```',
                                 '~~~py\narr[1]\n~~~', '```py\narr[1]', '~~~~py\narr[1]\n~~~\narr[2]'])
def test_mask_code_closed_and_unclosed(cfg, code):
    text = f'事实[1]。\n{code}'
    masked = mask_code(text)
    assert len(masked) == len(text)
    assert [i for i, ch in enumerate(masked) if ch == '\n'] == [i for i, ch in enumerate(text) if ch == '\n']
    assert [c['n'] for s in parse(text, cfg) for c in s.cites] == [1]


@pytest.mark.parametrize('heading', ['**补充说明（以下内容不来自资料）**', '### 补充说明(以下内容不来自资料)',
                                   '# 补充说明（以下内容不来自资料）  '])
def test_supplement_heading_and_priority(cfg, heading):
    text = f'事实[1]。\n{heading}\n未知[99]。'
    sents = parse(text, cfg)
    assert sents[-1].cites[0]['state'] == 'in_supplement'
    assert find_supplement(mask_code(text), cfg.prompt.supplement_heading) == len('事实[1]。\n')
    assert find_supplement('这是' + heading, cfg.prompt.supplement_heading) is None


@pytest.mark.parametrize('text', ['句子。[1]第二句[1]。', '句子[1]。第二句。[1]'])
def test_citations_after_punctuation(cfg, text):
    sents = parse(text, cfg)
    assert len(sents) == 2
    assert [len(s.cites) for s in sents] == [1, 1]
    assert sents[0].claim == '句子。'


def test_invalid_number(cfg):
    sents = parse('未知[99]；已知[1]。', cfg)
    assert [c['state'] for s in sents for c in s.cites] == ['invalid_number', 'unverified']


def test_timeout_keeps_unverified_and_final(cfg):
    cfg = cfg.model_copy(update={'postprocess': cfg.postprocess.model_copy(update={'verify_timeout_s': .005})})

    async def slow(pairs):
        await asyncio.sleep(1)
        return [0.] * len(pairs)

    state = dict(intent='rag', question='端口？', answer='服务默认使用端口19530[1]。',
                 context_blocks=[block().model_dump()], gen_meta={'status': 'ok'})
    final, plain, stats = asyncio.run(build_final(state, cfg, slow))
    assert final['type'] == 'final' and final['notice'] is None
    assert final['citations'][0]['state'] == 'unverified'
    assert stats['verify_degraded'] is True
    assert plain == '服务默认使用端口19530。'


@pytest.mark.parametrize('scores', [[.8], [.8, float('nan')], [.8, 'bad'], [.8, 1.1], [.8, True]])
def test_bad_batch_never_partially_mutates(cfg, scores):
    sents = parse('服务默认使用端口19530[1][1]。', cfg)
    before = copy.deepcopy(sents)

    async def bad(pairs):
        return scores

    with pytest.raises(VerifyDegraded):
        asyncio.run(check_support(sents, {1: block()}, {}, bad, cfg.postprocess))
    assert sents == before


def test_plain_preserves_code_and_incomplete(cfg):
    answer = '正文[99]。`arr[1]`\n```py\narr[1]\n```\n补充[1]。'
    plain = make_plain(answer, parse(answer, cfg), 'truncated', cfg.postprocess)
    assert plain == '正文。`arr[1]`\n```py\narr[1]\n```\n补充。' + cfg.postprocess.incomplete_suffix


def test_utf16_emoji_and_supplement(cfg):
    text = '😀事实[1]。\n' + cfg.prompt.supplement_heading + '\n😁[9]'

    async def score(pairs):
        return [.8] * len(pairs)

    final, _, _ = asyncio.run(build_final(dict(intent='rag', answer=text, context_blocks=[block().model_dump()]), cfg, score))
    assert final['citations'][0]['start'] == 4
    assert final['citations'][0]['end'] == 7
    pos = text.index('[9]')
    assert final['citations'][1]['start'] == pos + 2
    assert final['supplement']['start'] == utf16_offset(text, text.index('**'))


def test_windows_best_score_pairs_limit_and_warning(cfg):
    sents = parse('服务默认使用端口19530[1]。另一个默认端口是5432[1]。', cfg)
    chunks = {'c1': '不相关的内容', 'c2': '服务默认使用端口19530。'}
    seen = []

    async def score(pairs):
        seen.extend(pairs)
        return [.1, .1, .9, .1]

    asyncio.run(check_support(sents, {1: block()}, chunks, score, cfg.postprocess))
    assert len(seen) == 4
    assert sents[0].cites[0]['score'] == .9
    assert sents[1].cites[0]['state'] == 'unsupported'
    assert not support_warning(sents, cfg.postprocess)  # 恰好一半，不是超过
    sents[0].cites[0]['state'] = 'unverified'
    assert support_warning(sents, cfg.postprocess)  # unverified 不进入分母
    cfg = cfg.model_copy(update={'postprocess': cfg.postprocess.model_copy(update={'max_pairs': 1})})
    sents = parse('服务默认使用端口19530[1]。另一个默认端口是5432[1]。', cfg)

    async def one(pairs):
        assert len(pairs) == 1
        return [.9]

    asyncio.run(check_support(sents, {1: block()}, chunks, one, cfg.postprocess))
    assert sents[0].cites[0]['state'] == 'ok' and sents[1].cites[0]['state'] == 'unverified'
    assert ''.join(token_windows('abcdef', 2, len)) == 'abcdef'
    assert all(len(w) <= 2 for w in token_windows('abcdef', 2, len))


def test_one_supported_cite_supports_sentence(cfg):
    sents = parse('服务默认使用端口19530[1][2]。', cfg, {1: block(), 2: block(2)})

    async def score(pairs):
        return [.8, .1]

    asyncio.run(check_support(sents, {1: block(), 2: block(2)}, {}, score, cfg.postprocess))
    assert not support_warning(sents, cfg.postprocess)


@pytest.mark.parametrize('mode', ['chat', 'fallback', 'clarify'])
def test_non_rag_skips_verification(cfg, mode):
    async def forbidden(pairs):
        pytest.fail('不应调用重排')

    state = dict(intent='rag' if mode == 'fallback' else mode, answer='原文[1]', gen_meta={'scene': mode})
    final, plain, _ = asyncio.run(build_final(state, cfg, forbidden))
    assert final['sources'] == final['citations'] == []
    assert plain == state['answer']


def test_failed_has_no_final_or_plain(cfg):
    result = asyncio.run(build_final({'gen_meta': {'status': 'failed'}}, cfg, None))
    assert result == (None, None, {})


def test_sources_sorted_only_used_and_answer_never_changed(cfg):
    state = dict(intent='rag', answer='服务默认使用端口19530[3][1][99]。',
                 context_blocks=[block(3).model_dump(), block(1).model_dump(), block(2).model_dump()])
    before = copy.deepcopy(state)

    async def score(pairs):
        return [.1] * len(pairs)

    final, _, stats = asyncio.run(build_final(state, cfg, score))
    assert state == before
    assert [s['cite'] for s in final['sources']] == [1, 3]
    assert final['sources'][0]['pages'] == '2-3'
    assert final['notice'] == cfg.postprocess.notices.unsupported
    assert stats['warned']


@pytest.mark.parametrize('mark', ['[1]', '【1】'])
def test_compatible_citations_preserve_spans_and_code(cfg, mark):
    text = f'😀服务默认使用端口19530。{mark}[1]【1】第二句【99】。\n`arr[1]【1】`\n~~~text\n【1】\n~~~'
    sents = parse(text, cfg)
    cites = [c for s in sents for c in s.cites]
    assert [c['n'] for c in cites] == [1, 1, 1, 99]
    assert [text[c['start']:c['end']] for c in cites] == [mark, '[1]', '【1】', '【99】']
    assert sents[0].claim == '😀服务默认使用端口19530。'
    assert len(sents[0].cites) == 3
    assert cites[-1]['state'] == 'invalid_number'
    assert make_plain(text, sents, 'ok', cfg.postprocess).endswith('`arr[1]【1】`\n~~~text\n【1】\n~~~')


@pytest.mark.parametrize('mark', ['【1]', '[1】', '【1234】', '【 1 】', '【1,2】', '【1'])
def test_malformed_compatible_citations_not_recognized(cfg, mark):
    assert not any(s.cites for s in parse(f'事实{mark}。', cfg))


@pytest.mark.parametrize('score,expected', [(0., 'unsupported'), (.8, 'ok')])
def test_compatible_final_still_checks_support_and_preserves_answer(cfg, score, expected):
    answer = '😀服务默认使用端口19530【1】。\n' + cfg.prompt.supplement_heading + '\n补充【99】。`arr[1]【1】`'
    state = dict(intent='rag', answer=answer, context_blocks=[block().model_dump()])
    original = copy.deepcopy(state)
    calls = []

    async def rerank(pairs):
        calls.extend(pairs)
        return [score] * len(pairs)

    final, plain, stats = asyncio.run(build_final(state, cfg, rerank))
    assert state == original
    assert calls and all('【1】' not in claim for claim, _ in calls)
    first, supplement = final['citations']
    assert first['state'] == expected and first['score'] == score
    assert supplement['state'] == 'in_supplement'
    assert stats['citation_support'] == (1. if expected == 'ok' else 0.)
    assert final['notice'] == (cfg.postprocess.notices.unsupported if expected == 'unsupported' else None)
    assert final['supplement']['start'] == utf16_offset(answer, answer.index('**'))
    for c, mark in zip(final['citations'], ['【1】', '【99】'], strict=True):
        assert answer.encode('utf-16-le')[2*c['start']:2*c['end']].decode('utf-16-le') == mark
    assert plain == answer.replace('【1】', '', 1).replace('【99】', '')


def test_compatible_history_and_eval_cleaning(cfg):
    from ragagent.prompt.history import strip_old_citations
    from eval.ragas_eval import clean_answer

    answer = '事实[1]【2】。`arr[1]【2】`\n' + cfg.prompt.supplement_heading + '\n补充【3】。'
    assert strip_old_citations(answer) == '事实。`arr[1]【2】`\n' + cfg.prompt.supplement_heading + '\n补充。'
    assert clean_answer(answer, cfg.prompt.supplement_heading) == '事实。`arr[1]【2】`'


def test_support_claim_preserves_inline_code_and_ignores_code_citations(cfg):
    answer = '😀函数 `get_weather(city="北京")` 返回 `data[1]` 与 `status_code`【1】。'
    seen = []

    async def score(pairs):
        seen.extend(pairs)
        return [.8] * len(pairs)

    final, plain, _ = asyncio.run(build_final(
        dict(intent='rag', answer=answer, context_blocks=[block().model_dump()]), cfg, score))
    assert seen and seen[0][0] == '😀函数 get_weather(city="北京") 返回 data[1] 与 status_code。'
    assert len(final['citations']) == 1
    cite = final['citations'][0]
    assert answer.encode('utf-16-le')[2*cite['start']:2*cite['end']].decode('utf-16-le') == '【1】'
    assert plain == answer.replace('【1】', '')


def test_claim_excludes_fenced_code_but_keeps_inline_punctuation(cfg):
    answer = '若出现 `ToolError`，返回 `status="error"` 和 `str(e)`[1]。\n~~~py\nsecret[9]\n~~~\n`a; b[8]` 是表达式[1]。'
    sents = parse(answer, cfg)
    assert sents[0].claim == '若出现 ToolError，返回 status="error" 和 str(e)。'
    assert sents[-1].claim == 'a; b[8] 是表达式。'
    assert [c['n'] for s in sents for c in s.cites] == [1, 1]
    assert all('secret' not in s.claim for s in sents)


def test_claim_handles_multibacktick_and_multiline_inline_code(cfg):
    text = '表达式 ``a`b[7]`` 返回 `x_1\n+ y_2`[1]。'
    sents = parse(text, cfg)
    assert 'a`b[7]' in sents[0].claim
    assert 'x_1' in sents[0].claim
    assert sents[-1].claim == '+ y_2。'
    assert [c['n'] for s in sents for c in s.cites] == [1]


def test_inline_delimiters_do_not_pair_across_fenced_blocks(cfg):
    text = '未闭合的 `标记。\n~~~py\nsecret[9]\n~~~\n恢复正文 `status_code` 正常[1]。'
    sents = parse(text, cfg)
    assert sents[-1].claim == '恢复正文 status_code 正常。'
    assert all('secret' not in s.claim for s in sents)
    assert [c['n'] for s in sents for c in s.cites] == [1]


def test_verification_round_robin_and_incomplete_negative(cfg):
    limited = cfg.postprocess.model_copy(update={'max_pairs': 2})
    sents = parse('服务默认使用端口19530[1]。另一个默认端口是5432[1]。', cfg)
    seen = []
    async def scores(pairs):
        seen.extend(pairs)
        return [.1, .9]
    chunks = {'c1': '第一个窗口', 'c2': '第二个窗口'}
    asyncio.run(check_support(sents, {1: block()}, chunks, scores, limited))
    assert len(seen) == 2 and seen[0][0] != seen[1][0]
    assert sents[0].cites[0]['state'] == 'unverified'
    assert sents[1].cites[0]['state'] == 'ok'


def test_code_entity_from_question_qualifies_verification_claim(cfg):
    sents = parse('默认生成3个候选动作[1]。', cfg)
    seen = []
    async def scores(pairs):
        seen.extend(pairs)
        return [.8] * len(pairs)
    asyncio.run(check_support(sents, {1: block()}, {}, scores, cfg.postprocess,
                             question='expand_candidates默认生成几个候选动作？'))
    assert seen[0][0] == 'expand_candidates：默认生成3个候选动作。'
    assert sents[0].claim == '默认生成3个候选动作。'

@pytest.mark.parametrize('mark,kind', [
    ('【1:手册】', 'number_with_title'),
    ('【手册:】', 'title_without_number'),
    ('citeturn1', 'foreign_marker'),
])
def test_malformed_citations_are_reported_without_mapping(cfg, mark, kind):
    from ragagent.postprocess.parse import citation_format_issues
    text = f'事实{mark}。`{mark}`\n```text\n{mark}\n```'
    issues = citation_format_issues(mask_code(text), {1: block()})
    assert issues == [dict(kind=kind, text=mark, start=2, end=2 + len(mark))]
    assert not any(s.cites for s in parse(text, cfg))


def test_ordinary_brackets_and_valid_numbers_are_not_syntax_issues():
    from ragagent.postprocess.parse import citation_format_issues
    assert citation_format_issues('【重点】、[1]、【1】、【普通说明】', {1: block()}) == []


@pytest.mark.parametrize('answer,expected', [
    ('资料未提供该信息。', 'missing'),
    ('事实【1:手册】。', 'invalid_syntax'),
    ('事实[99]。', 'invalid_number'),
    ('事实[1]。', 'ok'),
])
def test_final_distinguishes_citation_format_states(cfg, answer, expected):
    async def scores(pairs):
        return [1.] * len(pairs)
    event, _, stats = asyncio.run(build_final(dict(answer=answer, context_blocks=[block().model_dump()],
        gen_meta=dict(scene='rag', status='ok')), cfg, scores))
    assert stats['citation_format_status'] == expected
    assert bool(stats['citation_format_issues']) == (expected == 'invalid_syntax')
    assert all(c['n'] != 99 or c['state'] == 'invalid_number' for c in event['citations'])
