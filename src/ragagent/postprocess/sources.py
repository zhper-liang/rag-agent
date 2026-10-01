def pages(block) -> str | None:
    if block.page_start < 0:
        return None
    return str(block.page_start) if block.page_start == block.page_end else f'{block.page_start}-{block.page_end}'


def build_sources(sents, blocks) -> list[dict]:
    used = {c['n'] for s in sents for c in s.cites
            if c['state'] not in ('invalid_number', 'in_supplement')}
    out = []
    for n in sorted(used):
        block = blocks[n]
        item = dict(cite=n, title=block.title, section=block.section_path, doc_id=block.doc_id,
                    images=[img.model_dump() for img in block.images])
        if (value := pages(block)) is not None:
            item['pages'] = value
        out.append(item)
    return out
