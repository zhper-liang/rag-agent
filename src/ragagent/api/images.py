"""清洗图片路径→前端URL；禁止把本地文件系统路径发给客户端。"""
from __future__ import annotations

import re
from pathlib import PurePosixPath
from urllib.parse import quote

IMAGE_PATH = re.compile(r'clean/([0-9a-f]{16})/v([1-9]\d*)/images/([^/]+)')
EXTENSIONS = frozenset({'.png', '.jpg', '.jpeg', '.gif', '.webp'})


def safe_filename(filename: str) -> bool:
    return (bool(filename) and len(filename) <= 255 and '\\' not in filename and '/' not in filename
            and '\x00' not in filename and filename not in ('.', '..')
            and PurePosixPath(filename).suffix.lower() in EXTENSIONS)


def image_url(image_path: str) -> str | None:
    m = IMAGE_PATH.fullmatch(image_path)
    if m is None or not safe_filename(m[3]):
        return None
    return f'/api/images/{m[1]}/v{m[2]}/{quote(m[3], safe="")}'


def public_image(image: dict) -> dict:
    out = {k: v for k, v in image.items() if k != 'image_path'}
    out['url'] = image_url(image.get('image_path', ''))
    return out


def public_event(event: dict) -> dict:
    if event['type'] == 'meta':
        return dict(event, attachments=[public_image(i) for i in event.get('attachments', [])])
    if event['type'] == 'final':
        return dict(event, sources=[dict(s, images=[public_image(i) for i in s.get('images', [])])
                                    for s in event.get('sources', [])])
    return event
