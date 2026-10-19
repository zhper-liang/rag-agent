"""文档切分入口：run_chunk(CleanDocument) → list[Chunk]（§15）。"""
import argparse
import json
from pathlib import Path
from statistics import median

from ragagent.config import load_config
from ragagent.ingest.models import CleanDocument
from .models import Chunk, make_chunk_id
from .tree import build_tree
from .units import prepare_blocks
from .packer import Packer
from .images import attach_images
from .dedup import content_hash, mark_duplicates


def _embed_text(draft, title, cfg):
    path = [title]+draft.section_path
    if len(path) > cfg.path_max_depth:
        path = [title]+path[-(cfg.path_max_depth-1):] if cfg.path_max_depth>1 else [title]
    prefix = ' > '.join(x[:cfg.path_heading_chars] for x in path)
    extra = []
    if draft.meta.get('lead_in'):
        extra.append(draft.meta['lead_in'])
    if draft.meta.get('split_part'):
        extra.append('(续 '+draft.meta['split_part']+')')
    if draft.meta.get('strategy') == 'split':
        for b in draft.blocks:
            caption = b.meta.get('table_caption')
            if caption:
                extra.append(caption)
    body = draft.text
    lines = body.splitlines(keepends=True)
    if draft.section_path and lines and lines[0].lstrip('#').strip() == draft.section_path[-1]:
        body = ''.join(lines[1:]).lstrip('\n')
    return '\n'.join([prefix]+extra+[body])


def _service_encode(texts):
    # 懒调用，仅真正需要语义断点时访问服务；默认路径不静默关闭语义。
    from ragagent.index.embedder import Embedder
    with Embedder() as emb:
        return emb.text(texts)


def run_chunk(doc: CleanDocument, *, cfg=None, count=None, encode=None) -> list[Chunk]:
    """可注入 cfg/count/encode 用于性质测试；生产默认使用 bge-m3 与配置。"""
    cfg = cfg or load_config().chunking
    if count is None:
        from .tokenizer import count_tokens
        count = count_tokens
    if not 0 < cfg.target_tokens <= cfg.max_tokens:
        raise ValueError('必须满足 0 < target_tokens <= max_tokens')
    if encode is None and cfg.semantic.enabled:
        encode = _service_encode
    blocks = prepare_blocks(doc.blocks)
    root = build_tree(blocks,count)
    headings = {b.meta['_section_idx']:b for b in blocks if b.type=='heading'}
    drafts = Packer(root,headings,cfg,count,encode).chunk_section(root)
    chunks = []
    for seq,d in enumerate(drafts):
        pages = [b.page for b in d.blocks if b.page is not None]
        chunks.append(Chunk(chunk_id=make_chunk_id(doc.doc_id,doc.version,seq),
            doc_id=doc.doc_id,version=doc.version,seq=seq,type=d.type,text=d.text,
            embed_text=_embed_text(d,doc.title,cfg),content_hash=content_hash(d.text),tokens=count(d.text),
            section_path=d.section_path,section_idx=d.section_idx,section_tokens=d.section_tokens,
            page_start=min(pages) if pages else None,page_end=max((b.meta.get('page_end', b.page) for b in d.blocks if b.page is not None), default=None),meta=d.meta))
    for i,c in enumerate(chunks):
        c.prev_id = chunks[i-1].chunk_id if i else None
        c.next_id = chunks[i+1].chunk_id if i+1 < len(chunks) else None
        if c.type == 'code' and c.meta.get('lead_in') and i and chunks[i-1].type == 'text' and chunks[i-1].section_idx == c.section_idx:
            c.related_ids.append(chunks[i-1].chunk_id)
            chunks[i-1].related_ids.append(c.chunk_id)
    attach_images(chunks,blocks,doc,cfg,count)
    return mark_duplicates(chunks)


def restore_text(chunks) -> str:
    """仅用正文及字符区间移除生成字符、图片占位符和 overlap，复原 Markdown。"""
    return ''.join(c.text[a:b] for c in sorted(chunks,key=lambda c:c.seq) if c.type != 'image'
                   for a,b in c.meta['restore_ranges'])


def statistics(chunks,cfg):
    text_chunks = [c for c in chunks if c.type!='image']
    sizes = sorted(c.tokens for c in text_chunks)
    strategies = {}
    for c in chunks:
        key = c.meta.get('strategy','unknown')
        strategies[key] = strategies.get(key,0)+1
    n = len(sizes)
    return {'chunks':len(chunks),'p50':median(sizes) if n else 0,
        'p90':sizes[min(n-1,int(n*0.9))] if n else 0,'max':max(sizes,default=0),
        'strategies':strategies,'small_ratio':sum(x<cfg.min_tokens for x in sizes)/n if n else 0,
        'split_ratio':sum(c.meta.get('strategy')=='split' for c in text_chunks)/n if n else 0}


def main():
    parser = argparse.ArgumentParser(description='切分 clean.json 并输出 Chunk JSONL')
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    from ragagent.ingest.lock import IngestBusy, ingest_lock
    cfg = load_config()
    try:
        with ingest_lock(cfg.paths.ingest_lock):
            doc = CleanDocument.model_validate_json(args.input.read_text())
            chunks = run_chunk(doc, cfg=cfg.chunking)
            output = args.output or Path(cfg.paths.chunks) / doc.doc_id / f'v{doc.version}.jsonl'
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(''.join(c.model_dump_json()+'\n' for c in chunks))
            print(json.dumps(statistics(chunks,cfg.chunking),ensure_ascii=False))
    except IngestBusy as exc:
        raise SystemExit(str(exc)) from None


if __name__ == '__main__':
    main()
