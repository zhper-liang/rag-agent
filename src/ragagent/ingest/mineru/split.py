"""云端200页/200MB限额预处理与页码偏移合并；不修改原始文件。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import pymupdf
from pydantic import BaseModel, Field

from ragagent.ingest.dedup import sha256
from ragagent.ingest.images import copy_image, local_image

MAX_PAGES = 200
MAX_BYTES = 200*1024*1024


class PDFPart(BaseModel):
    path: Path
    start_page: int = Field(ge=0)
    page_count: int = Field(gt=0)


def split_pdf(path: str | Path, out_dir: str | Path, *, max_pages=MAX_PAGES, max_bytes=MAX_BYTES) -> list[PDFPart]:
    path, out_dir = Path(path), Path(out_dir)
    if max_pages < 1 or max_bytes < 1:
        raise ValueError('分页/容量限制必须>0')
    with pymupdf.open(path) as document:
        if document.needs_pass or document.page_count < 1:
            raise ValueError('无法解析加密/空PDF')
        if document.page_count <= max_pages and path.stat().st_size <= max_bytes:
            return [PDFPart(path=path, start_page=0, page_count=document.page_count)]
        out_dir.mkdir(parents=True, exist_ok=True)
        source_hash = sha256(path)
        manifest = out_dir / '.parts.json'
        try:
            cached = json.loads(manifest.read_text())
            if cached['source_hash'] == source_hash and cached['max_pages'] == max_pages and cached['max_bytes'] == max_bytes:
                found = []
                for info in cached['parts']:
                    target = out_dir/info['name']
                    if target.is_symlink() or not target.resolve().is_relative_to(out_dir.resolve()) or sha256(target) != info['hash']:
                        raise ValueError('分段缓存无效')
                    found.append(PDFPart(path=target,start_page=info['start'],page_count=info['count']))
                if found:
                    return found
        except (OSError, ValueError, KeyError, TypeError):
            pass
        parts, start = [], 0
        while start < document.page_count:
            count = min(max_pages, document.page_count-start)
            while True:
                with pymupdf.open() as part:
                    part.insert_pdf(document, from_page=start, to_page=start+count-1)
                    data = part.tobytes(garbage=4, deflate=True)
                if len(data) <= max_bytes:
                    break
                if count == 1:
                    raise ValueError('单页PDF仍超过云端大小限制，无法按页继续拆分')
                count = max(1, count//2)
            target = out_dir/f'part-{start:06d}-{start+count:06d}.pdf'
            if target.is_symlink():
                raise ValueError('拆分输出不允许symlink')
            target.write_bytes(data)
            parts.append(PDFPart(path=target, start_page=start, page_count=count))
            start += count
        manifest.write_text(json.dumps({'source_hash': source_hash, 'max_pages': max_pages, 'max_bytes': max_bytes,
            'parts': [{'name':p.path.name,'start':p.start_page,'count':p.page_count,'hash':sha256(p.path)} for p in parts]}))
        return parts


def office_to_pdf(path: str | Path, out_dir: str | Path, *, executable='libreoffice', timeout=600) -> Path:
    path, out_dir = Path(path).resolve(), Path(out_dir).resolve()
    if path.suffix.lower() not in {'.doc', '.docx', '.ppt', '.pptx'}:
        raise ValueError('Office预转换仅支持Word/PPT')
    if shutil.which(executable) is None:
        raise FileNotFoundError('缺少LibreOffice，无法预转换超大Office')
    out_dir.mkdir(parents=True, exist_ok=True)
    # 独立profile避免多任务抢占正在打开的办公应用，不接受shell参数拼接。
    with tempfile.TemporaryDirectory(prefix='l1-libreoffice-') as profile:
        result = subprocess.run([executable, '--headless', '-env:UserInstallation='+Path(profile).as_uri(),
            '--convert-to', 'pdf', '--outdir', str(out_dir), str(path)], capture_output=True, timeout=timeout)
    target = out_dir/(path.stem+'.pdf')
    if result.returncode or not target.is_file() or target.is_symlink():
        raise RuntimeError('LibreOffice转换失败或未生成PDF')
    return target


def merge_parts(parts: list[tuple[PDFPart, Path]], out_dir: str | Path, *, source_path: str | Path) -> Path:
    """聚合content_list，page_idx累加偏移，截图改为安全无损hash副本。"""
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.is_symlink():
        raise ValueError('合并输出不允许symlink')
    stage = Path(tempfile.mkdtemp(prefix='.merge-', dir=out_dir.parent))
    items = []
    try:
        expected = 0
        for part, parsed in sorted(parts, key=lambda pair: pair[0].start_page):
            if part.start_page != expected:
                raise ValueError('PDF分段存在缺页/重叠')
            expected += part.page_count
            lists = sorted(Path(parsed).rglob('*content_list.json'))
            if len(lists) != 1:
                raise ValueError('分段content_list不唯一')
            values = json.loads(lists[0].read_text(encoding='utf-8'))
            for value in values:
                value = dict(value)
                if value.get('page_idx') is not None:
                    idx = int(value['page_idx'])
                    if not 0 <= idx < part.page_count:
                        raise ValueError('分段页码越界')
                    value['page_idx'] = idx+part.start_page
                for key in ('img_path', 'image_path'):
                    if value.get(key):
                        image = local_image(value[key], base=lists[0].parent, root=Path(parsed))
                        value[key] = copy_image(image, stage)
                items.append(value)
        (stage/'merged_content_list.json').write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')
        # 用同一映射渲染，不把原分段full.md中的旧图片路径带到输出。
        from ragagent.ingest.mineru.convert import content_list_to_blocks
        from ragagent.ingest.pipeline import render_markdown
        (stage/'full.md').write_text(render_markdown(content_list_to_blocks(items)), encoding='utf-8')
        manifest = {'source_sha256': sha256(source_path), 'files': {
            p.relative_to(stage).as_posix(): sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()}}
        (stage/'_complete.json').write_text(json.dumps(manifest), encoding='utf-8')
        if out_dir.exists():
            shutil.rmtree(out_dir)
        stage.replace(out_dir)
        return out_dir
    finally:
        if stage.exists():
            shutil.rmtree(stage)
