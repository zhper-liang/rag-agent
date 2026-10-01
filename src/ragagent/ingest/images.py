"""图片路径隔离、无损复制、pHash 过滤及图文上下文。"""
from collections import defaultdict
import hashlib
import logging
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote, urlsplit

import imagehash
from PIL import Image, ImageStat, UnidentifiedImageError

from ragagent.config import load_config
from ragagent.ingest.models import Block

log = logging.getLogger(__name__)
RASTER = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tif', '.tiff'}


def local_image(source: str, *, base: Path, root: Path) -> Path:
    """只读输入根下的相对位图，不请求远端、不读取任意本机文件。"""
    url = urlsplit(source)
    name = unquote(url.path)
    p = PurePosixPath(name)
    if url.scheme or url.netloc or not name or p.is_absolute() or '\\' in name or '\x00' in name:
        raise ValueError('图片不是安全的本地相对路径')
    candidate = (base / name).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError('图片越界或不存在')
    if candidate.suffix.lower() not in RASTER:
        raise ValueError('不支持的图片格式')
    return candidate


def copy_image(source: Path, out_dir: Path) -> str:
    """目标为内容hash文件名；保留完整原图，防止同名覆盖。"""
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    directory = out_dir / 'images'
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.resolve().is_relative_to(out_dir.resolve()):
        raise ValueError('图片输出路径越界')
    destination = directory / (digest + source.suffix.lower())
    if destination.is_symlink():
        raise ValueError('图片目标不允许symlink')
    # 同一次读取决定hash和复制内容，避免源文件并发变化造成名称与字节不一致。
    destination.write_bytes(content)
    return 'images/' + destination.name


def process_images(blocks: list[Block], out_dir: Path, *, cfg=None) -> list[Block]:
    cfg = cfg or load_config().ingest.images
    if cfg.llm_description:
        # 没有配置多模态描述接入点，不把开关当成已实现的能力。
        raise NotImplementedError('图片LLM描述尚未接入；请保持llm_description=false')
    pages = {b.page for b in blocks if b.page is not None}
    hashes, repeated = {}, defaultdict(set)
    for i, block in enumerate(blocks):
        if block.type != 'image' or not block.image_path:
            continue
        try:
            path = local_image(block.image_path, base=out_dir, root=out_dir)
            with Image.open(path) as image:
                w, h = image.size
                # 对灰度亮度求标准差，纯色(包括彩色)也能被正确排除。
                std = ImageStat.Stat(image.convert('L')).stddev[0]
                if min(w, h) < cfg.min_side_px or w*h < cfg.min_area_px or std < 2:
                    continue
                phash = str(imagehash.phash(image))
            hashes[i] = phash
            if block.page is not None:
                repeated[phash].add(block.page)
        except (OSError, ValueError, UnidentifiedImageError):
            log.warning('丢弃不可用的图片 block=%s', i)
    seen, result = set(), []
    for i, block in enumerate(blocks):
        b = block.model_copy(deep=True)
        if b.type != 'image':
            result.append(b)
            continue
        phash = hashes.get(i)
        # 一页文档不能把所有图片当作页眉；至少跨两个页面重复。
        if phash is None or phash in seen or (len(repeated[phash]) > 1 and pages and
                len(repeated[phash])/len(pages) > cfg.repeat_page_ratio):
            continue
        seen.add(phash)
        context = b.text.strip()
        if not context:
            before = next((x.text for x in reversed(blocks[:i]) if x.type == 'paragraph'), '')
            after = next((x.text for x in blocks[i+1:] if x.type == 'paragraph'), '')
            context = '\n'.join(t[:cfg.context_chars] for t in (before, after) if t)
        b.meta.update(phash=phash, context=context)
        result.append(b)
    return result


def prune_images(blocks: list[Block], out_dir: Path):
    used = {b.image_path for b in blocks if b.image_path}
    for image in (out_dir / 'images').glob('*'):
        if image.is_file() and ('images/'+image.name) not in used:
            image.unlink()
