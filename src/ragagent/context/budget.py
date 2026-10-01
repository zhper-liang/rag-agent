"""P1/P2/P3 原子填充：整块取舍，不截断原文；附件为 P4 零预算。"""

from __future__ import annotations

import logging
from math import ceil

from ragagent.context.images import dedup_images, image_available, resolve_image_path
from ragagent.context.merge import chunk_groups, rank_score

log = logging.getLogger(__name__)


def chunk_cost(c: dict, cfg) -> int:
    if c["tokens"] < 0 or cfg.context.token_safety_factor <= 0:
        raise ValueError("tokens 不得为负，token_safety_factor 必须大于零")
    return ceil(c["tokens"] * cfg.context.token_safety_factor)


def context_usage(chosen, roles, send_image, cfg) -> dict[str, int]:
    usage = {"P1": 0, "P2": 0, "P3": 0, "P4": 0}
    for cid, c in chosen.items():
        priority = "P1" if roles[cid] == "hit" else ("P3" if roles[cid] in {"section", "neighbor"} else "P2")
        usage[priority] += chunk_cost(c, cfg)
    usage["P1"] += len(send_image) * cfg.context.image.tokens_per_image
    usage["overhead"] = len(chunk_groups(chosen, roles)) * cfg.context.block_overhead_tokens
    usage["total"] = sum(usage.values())
    return usage


def fill(hits: list[dict], exp: dict[str, list[dict]], cfg):
    """返回 (chosen, roles, attachments, send_image)，exp 是契约中的扁平 role 列表。"""
    chosen, roles, send_image = {}, {}, set()
    ranked = sorted(hits, key=rank_score, reverse=True)

    def fits(proposed, proposed_roles, images):
        # 预扣 top_n；超过 top_n 的伴随段落另外扣，不能违反总成本不变量。
        blocks = max(cfg.rerank.top_n, len(chunk_groups(proposed, proposed_roles)))
        total = (sum(chunk_cost(c, cfg) for c in proposed.values())
                 + len(images) * cfg.context.image.tokens_per_image
                 + blocks * cfg.context.block_overhead_tokens)
        return total <= cfg.context.budget_tokens

    def take(items, role, image=None):
        proposed, proposed_roles = dict(chosen), dict(roles)
        for c in items:
            if c["chunk_id"] not in proposed:
                proposed[c["chunk_id"]], proposed_roles[c["chunk_id"]] = c, role
        images = send_image | ({image} if image else set())
        if not fits(proposed, proposed_roles, images):
            return False
        chosen.update(proposed)
        roles.update(proposed_roles)
        send_image.update(images)
        return True

    kept, seen = [], set()
    for h in ranked:
        cid = h["chunk_id"]
        if cid in seen:
            continue
        seen.add(cid)
        if h["type"] == "image":
            resolve_image_path(h["image_path"], cfg.paths.clean)
            if (len(send_image) < cfg.context.image.max_images
                    and image_available(h["image_path"], cfg.paths.clean)
                    and take([h], "hit", image=cid)):
                kept.append(h)
                continue
        if take([h], "hit"):
            kept.append(h)
        else:
            log.info("context hit omitted by budget")
    # 分批级别：所有 P1 优先于任何 P2，所有 P2 优先于任何 P3。
    for h in kept:
        for role in ("lead_in", "split_sibling", "anchor"):
            for c in exp.get(h["chunk_id"], []):
                if c["role"] == role:
                    take([c], role)
    for h in kept:
        if h["type"] == "image":
            continue
        items = exp.get(h["chunk_id"], [])
        sec = [c for c in items if c["role"] == "section" and c["type"] != "image"]
        # 阈值双重检查，调用者即使错误提供大节也不能吞掉预算。
        if h["section_tokens"] <= cfg.expansion.section_max_tokens and sec and take(sec, "section"):
            continue
        # 小节原子取不下时，从已取回整节中找到相邻块；不新增查询。
        neighbors = {c["chunk_id"]: c for c in items if c["role"] == "neighbor"}
        neighbors.update({c["chunk_id"]: c for c in sec if c["chunk_id"] in {h["prev_id"], h["next_id"]}})
        for c in sorted(neighbors.values(), key=lambda c: c["seq"]):
            take([c], "neighbor")
    attachments = dedup_images(exp, kept, cfg)
    assert context_usage(chosen, roles, send_image, cfg)["total"] <= cfg.context.budget_tokens
    return chosen, roles, attachments, send_image
