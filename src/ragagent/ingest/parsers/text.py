"""TXT 空行分段；严格 UTF-8，错误编码交给批次失败处理。"""
from pathlib import Path
import re
from ragagent.ingest.models import Block


def parse_text(path: str | Path) -> list[Block]:
    text = Path(path).read_text(encoding='utf-8-sig').replace('\r\n', '\n')
    return [Block(type='paragraph', text=p.strip()) for p in re.split(r'\n\s*\n', text) if p.strip()]
