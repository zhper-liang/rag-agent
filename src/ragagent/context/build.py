"""上下文入口，返回 ContextBlock 与展示附件（上下文构造 §12）。"""

from ragagent.context.budget import context_usage, fill
from ragagent.context.merge import merge_blocks
from ragagent.retrieval.models import Expansion, Hit


def build_context_details(hits, expansions, cfg, *, metrics=None):
    valid_hits = [Hit.model_validate(h).model_dump() for h in hits]
    valid_exp = {cid: [Expansion.model_validate(c).model_dump() for c in values]
                 for cid, values in expansions.items()}
    chosen, roles, attachments, send_image = fill(valid_hits, valid_exp, cfg, metrics=metrics)
    blocks = merge_blocks(chosen, roles, send_image, attachments)
    ids = [cid for block in blocks for cid in block.chunk_ids]
    assert len(ids) == len(set(ids)), "同一个 chunk 不得出现多次"
    return blocks, attachments, context_usage(chosen, roles, send_image, cfg)


def build_context(hits, expansions, cfg):
    blocks, attachments, _ = build_context_details(hits, expansions, cfg)
    return blocks, attachments
