"""图片路径再校验；原图只来自 data/clean，解码失败退回说明文字。"""

from __future__ import annotations

import io
import logging
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps

from ragagent.context.models import ImageRef

log = logging.getLogger(__name__)


def resolve_image_path(image_path: str, clean_root: str | Path = "data/clean") -> Path:
    """image_path 是相对于 data 的 clean/...；同时阻止 symlink 越界。"""
    p = PurePosixPath(image_path)
    if (not image_path or p.is_absolute() or ".." in p.parts or "\\" in image_path
            or not p.parts or p.parts[0] != "clean" or len(p.parts) < 2):
        raise ValueError("image_path 必须在 clean/ 下且不含绝对路径或 ..")
    root = Path(clean_root).resolve()
    target = root.joinpath(*p.parts[1:]).resolve()
    if not target.is_relative_to(root):
        raise ValueError("image_path 越过 clean 根目录")
    return target


def image_available(image_path: str, clean_root: str | Path = "data/clean", *, metrics=None) -> bool:
    target = resolve_image_path(image_path, clean_root)
    try:
        with Image.open(target) as image:
            image.verify()
        return True
    except (OSError, ValueError, Image.DecompressionBombError):
        log.warning("image unavailable; use caption only")
        if metrics is not None:
            metrics.degraded.labels(kind="image").inc()
        return False


def prepare_image(image_path: str, cfg, *, metrics=None) -> bytes | None:
    """供 Prompt 读文件时使用：按 EXIF 转正、长边缩小、RGB JPEG；不写运行文件。"""
    target = resolve_image_path(image_path, cfg.paths.clean)
    try:
        with Image.open(target) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((cfg.context.image.max_side_px, cfg.context.image.max_side_px))
            out = io.BytesIO()
            image.save(out, format="JPEG", quality=cfg.context.image.jpeg_quality)
            return out.getvalue()
    except (OSError, ValueError, Image.DecompressionBombError):
        log.warning("image read failed; use caption only")
        if metrics is not None:
            metrics.degraded.labels(kind="image").inc()
        return None


def image_ref(chunk: dict, send_original: bool = False) -> ImageRef:
    return ImageRef(chunk_id=chunk["chunk_id"], image_path=chunk["image_path"],
                    caption=chunk["text"], send_original=send_original)


def dedup_images(exp: dict[str, list[dict]], kept: list[dict], cfg) -> list[ImageRef]:
    """仅保留已保留命中的展示附件；命中原图不消耗附件限额。"""
    out, seen = [], {h["chunk_id"] for h in kept if h["type"] == "image"}
    for h in kept:
        count = 0
        for c in exp.get(h["chunk_id"], []):
            if c["role"] != "attachment" or c["type"] != "image":
                continue
            if c["chunk_id"] in seen:
                continue
            resolve_image_path(c["image_path"], cfg.paths.clean)
            if count >= cfg.expansion.images_per_chunk or len(out) >= cfg.expansion.images_total:
                break
            out.append(image_ref(c))
            seen.add(c["chunk_id"])
            count += 1
    return out
