"""§9 标题层级优先，其次节内语义/贪心，最后同节小块收尾。"""
import re

from .models import Draft, Unit
from .semantic import boundaries
from .splitter import Fragment, split_unit, whole, hard_cut
from .units import build_units, render_block, unit_text


def combine(fragments):
    text, blocks, spans, ranges = '', [], [], []
    for f in fragments:
        offset = len(text)
        text += f.text
        blocks.extend(f.blocks)
        spans.extend(f.source_spans)
        ranges.extend((a+offset,b+offset) for a,b in f.restore_ranges)
    return Fragment(text=text,blocks=blocks,source_spans=spans,restore_ranges=ranges)


class Packer:
    def __init__(self, root, headings, cfg, count, encode=None):
        self.root, self.headings, self.cfg, self.count, self.encode = root, headings, cfg, count, encode

    def blocks(self, sec):
        if not sec.body and not any(self.blocks(c) for c in sec.children):
            return []
        out = ([self.headings[sec.idx]] if sec.heading is not None else []) + sec.body
        return out + [b for child in sec.children for b in self.blocks(child)]

    def draft(self, f, sec, strategy):
        meta = {'strategy':strategy, **f.meta, 'source_spans':f.source_spans,
                'restore_ranges':f.restore_ranges,
                'source_block_indices':list(dict.fromkeys(b.meta['_source_idx'] for b in f.blocks))}
        if f.kind == 'code':
            meta.setdefault('lang',f.blocks[0].meta.get('lang',''))
        screenshots = [b.image_path for b in f.blocks if b.type in ('table','equation') and b.image_path]
        if screenshots:
            meta['image_paths'] = screenshots
        return Draft(type='code' if f.kind == 'code' else 'text',text=f.text,blocks=f.blocks,
            tokens=self.count(f.text),section_path=sec.path,section_idx=sec.idx,
            section_tokens=sec.tokens,meta=meta)

    def has_hard_break(self, sec):
        return any(u.hard_break for u in build_units(self.blocks(sec),self.cfg,self.count))

    def fits_whole(self, sec):
        return bool(self.blocks(sec)) and sec.tokens <= self.cfg.target_tokens and not self.has_hard_break(sec)

    def whole_section(self, sec):
        f = whole(Unit(blocks=self.blocks(sec),kind='text',tokens=sec.tokens))
        return self.draft(f,sec,'section')

    def pack_units(self, blocks, sec):
        units = build_units(blocks,self.cfg,self.count)
        semantic = boundaries([unit_text(u) for u in units],self.cfg.semantic,self.encode) if sec.tokens > self.cfg.max_tokens else {}
        out, buf, previous_paragraph = [], [], None

        def flush():
            if buf:
                f = combine(buf)
                out.append(self.draft(f,sec,'semantic' if semantic else 'greedy'))
                buf.clear()
        for i,u in enumerate(units):
            parts = split_unit(u,self.cfg,self.count)
            if u.hard_break or any(f.meta.get('strategy') == 'split' for f in parts):
                flush()
                lead = ''
                if previous_paragraph is not None and (u.kind == 'code' or
                    (u.kind == 'list' and previous_paragraph.text.rstrip().endswith((':','：')))):
                    sentences = [s for s in re.split(r'(?<=[。！？!?])',previous_paragraph.text.strip()) if s]
                    if sentences:
                        lead = hard_cut(sentences[-1],self.cfg.lead_in_tokens,self.count)[0]
                for f in parts:
                    if lead:
                        f.meta['lead_in'] = lead
                    out.append(self.draft(f,sec,f.meta.get('strategy','code')))
            else:
                f = parts[0]
                candidate = ''.join(x.text for x in buf) + f.text
                size = self.count(''.join(x.text for x in buf))
                early = semantic.get(i) == 'break' and size >= self.cfg.target_tokens*self.cfg.semantic.early_flush_ratio
                strong = semantic.get(i) == 'strong'
                if buf and (early or self.count(candidate) > self.cfg.max_tokens or
                            (self.count(candidate) > self.cfg.target_tokens and not strong)):
                    flush()
                buf.append(f)
            previous_paragraph = u.blocks[-1] if u.blocks[-1].type == 'paragraph' else None
        flush()
        return self.merge_small(out)

    def merge_small(self, drafts):
        # split 的 overlap 必须留在头部，不与邻块混合；代码从不参与合并。
        def eligible(d):
            return d.type == 'text' and d.meta['strategy'] != 'split'
        i = 0
        while i < len(drafts):
            d = drafts[i]
            if not eligible(d) or d.tokens >= self.cfg.min_tokens:
                i += 1; continue
            j = None
            for candidate in (i-1,i+1):
                if 0 <= candidate < len(drafts):
                    other = drafts[candidate]
                    combined_text = other.text+d.text if candidate < i else d.text+other.text
                    if eligible(other) and other.section_idx == d.section_idx and self.count(combined_text) <= self.cfg.max_tokens:
                        j = candidate; break
            if j is None:
                i += 1; continue
            left,right = sorted((i,j))
            a,b = drafts[left],drafts[right]
            offset = len(a.text)
            a.text += b.text
            a.blocks += b.blocks
            a.tokens = self.count(a.text)
            a.meta['source_spans'] += b.meta['source_spans']
            a.meta['restore_ranges'] += [(x+offset,y+offset) for x,y in b.meta['restore_ranges']]
            a.meta['source_block_indices'] = list(dict.fromkeys(a.meta['source_block_indices']+b.meta['source_block_indices']))
            if 'image_paths' in b.meta:
                a.meta['image_paths'] = list(dict.fromkeys(a.meta.get('image_paths',[])+b.meta['image_paths']))
            drafts.pop(right)
            i = max(0,left-1)
        return drafts

    def chunk_section(self, sec):
        # 末尾无正文的孤立标题不输出；标题路径由树保留。
        if not self.blocks(sec):
            return []
        if not self.has_hard_break(sec) and (sec.tokens <= self.cfg.target_tokens or
                                           (not sec.children and sec.tokens <= self.cfg.max_tokens)):
            return [self.whole_section(sec)]
        body = ([self.headings[sec.idx]] if sec.heading is not None else []) + sec.body
        items = ([body] if body else []) + sec.children
        drafts, group = [], []

        def flush():
            if group:
                f = whole(Unit(blocks=[b for item in group for b in item],kind='text',tokens=0))
                drafts.append(self.draft(f,sec,'section'))
                group.clear()
        for item in items:
            is_body = isinstance(item,list)
            bs = item if is_body else self.blocks(item)
            size = self.count(''.join(render_block(b) for b in bs))
            hard = any(u.hard_break for u in build_units(bs,self.cfg,self.count))
            fits = size <= self.cfg.target_tokens and not hard
            candidate = ''.join(render_block(b) for g in group for b in g) + ''.join(render_block(b) for b in bs)
            if fits and self.count(candidate) <= self.cfg.target_tokens:
                group.append(bs); continue
            flush()
            if fits:
                group.append(bs)
            elif is_body:
                drafts.extend(self.pack_units(bs,sec))
            else:
                drafts.extend(self.chunk_section(item))
        flush()
        return drafts
