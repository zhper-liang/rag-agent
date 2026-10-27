"""离线重放已存答案的引用格式；不调用生成/重排模型，不冒充支持度复评。"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import re
from unittest.mock import patch

from ragagent.postprocess import parse
from ragagent.postprocess.offsets import utf16_citations
from ragagent.postprocess.plain import make_plain
from ragagent.prompt.history import strip_old_citations

LEGACY_CITE = re.compile(r"\[(\d{1,3})\]")


def replay(rows: list[dict], *, heading: str) -> dict:
    runs: dict[str, Counter] = {}
    changed = []
    for row in rows:
        state = row['state']
        answer = state['answer']
        blocks = {b['cite']: b for b in state['context_blocks']}
        parsed = []
        for legacy in (True, False):
            with patch.object(parse, 'CITE', LEGACY_CITE) if legacy else nullcontext():
                masked = parse.mask_code(answer)
                sents = parse.split_sentences(answer, masked, parse.find_supplement(masked, heading))
                parse.check_format(sents, blocks)
                parsed.append(sents)
        old, new = ([c for s in sents for c in s.cites] for sents in parsed)
        old_spans = {(c['start'], c['end']) for c in old}
        added = [c for c in new if (c['start'], c['end']) not in old_spans]
        assert all(c in new for c in old), '既有引用不可丢失或改变'
        for c in utf16_citations(answer, parsed[1]):
            token = answer.encode('utf-16-le', errors='surrogatepass')[2*c['start']:2*c['end']].decode('utf-16-le', errors='surrogatepass')
            assert parse.CITE.fullmatch(token), 'UTF-16 位置必须切出原始引用'
        # status=ok 不访问 incomplete_suffix；和历史兜底的去引用结果必须一致。
        assert make_plain(answer, parsed[1], 'ok', None) == strip_old_citations(answer)
        counts = runs.setdefault(row['run'], Counter())
        counts.update(answers=1, answerable=bool(row['case']['expect'].get('answerable')),
                      legacy_citations=len(old), compatible_citations=len(new),
                      recovered_zero_citation_answers=bool(not old and new), added_citations=len(added))
        for c in added:
            counts[f"added_{c['state']}"] += 1
        if added:
            changed.append(dict(run=row['run'], case_id=row['case']['id'],
                                legacy_citations=len(old), compatible_citations=len(new),
                                added_states=dict(Counter(c['state'] for c in added))))
        assert answer == state['answer']
    return dict(scope='format_only', support_revalidated=False,
                note='编号存在只表示 unverified；没有重排打分，不计算新的支持率或更新基线。',
                runs={run: dict(counts) for run, counts in runs.items()}, changed=changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--heading', required=True, help='使用 config.yaml 的 prompt.supplement_heading')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.states.read_text().splitlines() if line.strip()]
    report = replay(rows, heading=args.heading)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
