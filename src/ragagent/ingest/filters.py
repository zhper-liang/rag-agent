"""质量门槛，不把代码/公式中的合法符号当乱码。"""
import logging
import unicodedata
from ragagent.config import load_config
from ragagent.ingest.models import Block

log = logging.getLogger(__name__)


def garbled_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    # 替代字符、私用区、代理项、不可见控制、未分配字符才是明确乱码。
    invalid = sum(c == '\ufffd' or (unicodedata.category(c)[0] not in {'L', 'N', 'P', 'M'} and c not in '$=+*/<>%_`|~\\&^#') for c in chars)
    return invalid/len(chars)


def filter_quality(blocks: list[Block], *, cfg=None) -> list[Block]:
    cfg = cfg or load_config().ingest.quality
    result = []
    for b in blocks:
        if b.type not in {'heading', 'image'} and len(b.text.strip()) < cfg.min_block_chars:
            continue
        if garbled_ratio(b.text) > cfg.garbled_ratio:
            log.warning('丢弃乱码block type=%s page=%s', b.type, b.page)
            continue
        result.append(b.model_copy(deep=True))
    return result


def is_empty(blocks: list[Block], *, cfg=None) -> bool:
    cfg = cfg or load_config().ingest.quality
    return not any(b.type == 'image' for b in blocks) and sum(len(b.text.strip()) for b in blocks) < cfg.min_doc_chars
