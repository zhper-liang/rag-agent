"""按配置路由，不猜测二进制文件的内容。"""
from pathlib import Path
from ragagent.config import load_config


def route(path: str | Path, *, cfg=None) -> str | None:
    cfg = cfg or load_config().ingest.parse_routes
    ext = Path(path).suffix.lstrip('.').lower()
    if ext in cfg.text:
        return 'text'
    if ext in cfg.mineru:
        return 'mineru'
    return None
