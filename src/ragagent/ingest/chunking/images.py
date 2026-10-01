"""§10 图片锚点/反向关联；图片不进入正文 prev/next 链。"""
import re

from .models import Chunk, make_chunk_id
from .dedup import content_hash
from .splitter import hard_cut


def attach_images(chunks, blocks, doc, cfg, count):
    texts = [c for c in chunks if c.type == 'text']
    for b in blocks:
        if b.type != 'image':
            continue
        # 即便整节只有图片，也保留过占位符 text chunk，因此总有锚点。
        candidates = [c for c in texts if b.meta['_source_idx'] in c.meta['source_block_indices']]
        if not candidates:
            candidates = sorted(texts,key=lambda c:min(abs(b.meta['_source_idx']-i) for i in c.meta['source_block_indices']))
        if not candidates:
            raise ValueError('图片缺少文本锚点')
        anchor = candidates[0]
        # 去掉展示占位符，抽取前后句给图的编码文本。
        context = re.sub(r'\[图: [\s\S]*?\]\n\n','',anchor.text)
        sentences = [s.strip() for s in re.split(r'(?<=[。！？!?])',context) if s.strip()]
        caption = b.text
        context = b.meta.get('context') or (sentences[-1] if sentences else '')
        text = '\n'.join(dict.fromkeys(t for t in (caption, context) if t))
        text = hard_cut(text,cfg.max_tokens,count)[0] if text else ''
        cid = make_chunk_id(doc.doc_id,doc.version,len(chunks))
        image = Chunk(chunk_id=cid,doc_id=doc.doc_id,version=doc.version,seq=len(chunks),type='image',
            text=text,embed_text=' > '.join([doc.title]+anchor.section_path)+'\n'+text,
            content_hash=content_hash(text),tokens=count(text),section_path=anchor.section_path,
            section_idx=anchor.section_idx,section_tokens=anchor.section_tokens,
            page_start=b.page,page_end=b.page,image_path=b.image_path,
            meta={'anchor_id':anchor.chunk_id,'caption':caption,'strategy':'image'})
        image.related_ids = [anchor.chunk_id] + [x for x in (anchor.prev_id,anchor.next_id)
            if x is not None and next(c for c in chunks if c.chunk_id==x).section_idx == anchor.section_idx]
        anchor.image_ids.append(cid)
        chunks.append(image)
    return chunks
