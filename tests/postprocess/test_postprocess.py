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
        return [.1, .9, .1, .1]

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
