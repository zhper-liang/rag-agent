def make_plain(answer: str, sents, status: str, cfg) -> str:
    spans = sorted((c['start'], c['end']) for s in sents for c in s.cites)
    out, pos = [], 0
    for start, end in spans:
        out.append(answer[pos:start])
        pos = end
    out.append(answer[pos:])
    plain = ''.join(out)
    if status in ('truncated', 'interrupted'):
        plain += cfg.incomplete_suffix
    return plain
