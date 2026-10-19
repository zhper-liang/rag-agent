"""离线清洗入口：扫描/增量、解析、清洗、质量/去重、原子版本产物。"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import hashlib
import logging
from pathlib import Path
import re
import shutil
import tempfile

from ragagent.config import load_config
from ragagent.ingest.cleaners import clean_blocks
from ragagent.ingest.dedup import DocumentDedup, sha256
from ragagent.ingest.filters import filter_quality, is_empty
from ragagent.ingest.images import copy_image, process_images, prune_images
from ragagent.ingest.models import Block, CleanDocument
from ragagent.ingest.parsers.markdown import parse_markdown
from ragagent.ingest.parsers.text import parse_text
from ragagent.ingest.registry_db import RegistryDB, make_doc_id
from ragagent.ingest.router import route

log = logging.getLogger(__name__)


def render_markdown(blocks: list[Block]) -> str:
    rendered = []
    for b in blocks:
        if b.type == 'heading':
            value = '#'*min(6, max(1, b.level))+' '+b.text
        elif b.type == 'equation':
            value = '$$\n'+b.text+'\n$$'
        elif b.type == 'image':
            # full.md与clean.json同级，渲染的图片引用应使用images/而非data根路径。
            value = '!['+b.text.replace(']', r'\]')+'](images/'+Path(b.image_path).name+')' if b.image_path else b.text
        else:
            value = b.text
        if b.type == 'table' and b.meta.get('table_caption'):
            value = b.meta['table_caption']+'\n\n'+value
        if b.type == 'table' and b.meta.get('table_footnote'):
            value += '\n\n'+b.meta['table_footnote']
        if value:
            rendered.append(value)
    return '\n\n'.join(rendered)+'\n' if rendered else ''


def _safe_error(exc):
    # 不在登记表保存带签名查询参数或Bearer密钥的传输异常。
    text = re.sub(r'https?://\S+', '<url>', str(exc))
    text = re.sub(r'(?i)bearer\s+\S+', 'Bearer <redacted>', text)
    return type(exc).__name__+': '+text[:500]


def _prune_versions(root: Path, keep: int):
    entries = [(p, int(m[1])) for p in root.iterdir() if (m := re.fullmatch(r'\.?v([1-9]\d*)(?:-parts)?', p.name))] if root.exists() else []
    keep_numbers = set(sorted({n for _, n in entries}, reverse=True)[:keep])
    for p, number in entries:
        if number in keep_numbers:
            continue
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
            raise ValueError('旧版本路径越界')
        if p.is_dir():
            shutil.rmtree(p)


def _finish(path, record, cfg, db, dedup, *, parsed_dir=None):
    doc_id, version = record['doc_id'], record['version']
    if sha256(path) != record['content_hash']:
        raise ValueError('源文件在解析期间变化，拒绝写入错误版本')
    root = Path(cfg.paths.clean)
    parent = root/doc_id
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.resolve().is_relative_to(root.resolve()):
        raise ValueError('清洗输出目录越界')
    stage = Path(tempfile.mkdtemp(prefix='.clean-', dir=parent))
    destination = parent/f'v{version}'
    try:
        if parsed_dir is not None:
            from ragagent.ingest.mineru.convert import convert
            blocks = convert(parsed_dir, stage, max_cols=cfg.ingest.clean.table_max_cols)
            if record['file_type'] in {'png', 'jpg', 'jpeg'}:
                blocks.append(Block(type='image', text='', page=1, image_path=copy_image(path, stage)))
        elif path.suffix.lower() in {'.md', '.markdown'}:
            blocks = parse_markdown(path, stage, input_root=Path(record['_input_root']))
        else:
            blocks = parse_text(path)
        input_blocks = len(blocks)
        pages = max((b.page or 0 for b in blocks), default=0) or None
        if path.suffix.lower() == '.pdf':
            with path.open('rb') as source:
                pdf_header = source.read(5) == b'%PDF-'
            if pdf_header:
                import pymupdf
                with pymupdf.open(path) as source_pdf:
                    pages = source_pdf.page_count
        blocks = clean_blocks(blocks, cfg=cfg.ingest.clean, file_type=record['file_type'], filename=path.stem)
        blocks = process_images(blocks, stage, cfg=cfg.ingest.images)
        blocks = filter_quality(blocks, cfg=cfg.ingest.quality)
        title = next((b.text for b in blocks if b.type == 'heading' and b.level == 1), path.stem)
        full_text = render_markdown(blocks)
        if is_empty(blocks, cfg=cfg.ingest.quality):
            db.update(doc_id, status='empty', title=title, pages=pages, duplicate_of=None, error=None)
            return 'empty'
        # 仅正文参与近重复，路径/caption不计入，纯图片文档不做空文本MinHash。
        dedup_text = '\n'.join(b.text for b in blocks if b.type != 'image')
        duplicate = dedup.duplicate_of(dedup_text, exclude_doc_id=doc_id)
        if duplicate:
            db.update(doc_id, status='duplicate', title=title, pages=pages, duplicate_of=duplicate, error=None)
            return 'duplicate'
        prune_images(blocks, stage)
        for b in blocks:
            if b.image_path:
                b.image_path = f'clean/{doc_id}/v{version}/'+b.image_path
        doc = CleanDocument(doc_id=doc_id, version=version, source_path=record['source_path'],
            file_type=record['file_type'], parser=record['parser'], title=title,
            content_hash=record['content_hash'], blocks=blocks,
            meta={'input_blocks': input_blocks, 'retained_blocks': len(blocks), 'pages': pages})
        if sha256(path) != record['content_hash']:
            raise ValueError('源文件在清洗期间变化，拒绝写入错误版本')
        (stage/'clean.json').write_text(doc.model_dump_json(indent=2), encoding='utf-8')
        (stage/'full.md').write_text(full_text, encoding='utf-8')
        if destination.is_symlink():
            raise ValueError('版本输出不允许symlink')
        if destination.exists():
            shutil.rmtree(destination)
        stage.replace(destination)
        # 崩溃发生在rename之后、状态之前也可安全重试当前版本。
        dedup.remember(doc_id, version, dedup_text)
        db.update(doc_id, status='cleaned', title=title, pages=pages, duplicate_of=None, error=None)
        _prune_versions(parent, cfg.paths.keep_versions)
        _prune_versions(Path(cfg.paths.parsed)/doc_id, cfg.paths.keep_versions)
        return 'cleaned'
    finally:
        if stage.exists():
            shutil.rmtree(stage)


async def _cloud(jobs, cfg, db, resume, mineru):
    from ragagent.ingest.mineru.client import MinerUClient, ParseJob, _cached
    from ragagent.ingest.mineru.split import split_pdf, office_to_pdf, merge_parts, MAX_BYTES
    expanded, mapping, part_groups, prepared, results = [], {}, {}, {}, {}
    with db.transaction():
        db.conn.execute("""CREATE TABLE IF NOT EXISTS ingest_parse_parts(
            doc_id TEXT NOT NULL,version INTEGER NOT NULL,part_id TEXT NOT NULL,batch_id TEXT,
            PRIMARY KEY(doc_id,version,part_id))""")
    # 准备所有分段；一个输入预处理失败不影响其他文件。
    for job in jobs:
        try:
            if _cached(job):
                results[job.doc_id] = job.out_dir
                continue
            file = job.path
            workspace = job.out_dir.parent / ('.'+job.out_dir.name+'-parts')
            if not job.out_dir.parent.resolve().is_relative_to(Path(cfg.paths.parsed).resolve()) or workspace.is_symlink():
                raise ValueError('解析分段输出目录越界')
            with file.open('rb') as source:
                is_pdf = source.read(5) == b'%PDF-'
            converted = False
            if file.suffix.lower() in {'.doc', '.docx', '.ppt', '.pptx'} and file.stat().st_size > MAX_BYTES:
                file = office_to_pdf(file, workspace/'converted')
                is_pdf, converted = True, True
            if is_pdf:
                parts = split_pdf(file, workspace/'input')
            else:
                parts = []
            if len(parts) > 1 or converted or (parts and parts[0].path != job.path):
                group = []
                version = db.get(job.doc_id)['version']
                for part in parts:
                    part_id = hashlib.sha1(f'{job.doc_id}:{version}:{part.start_page}'.encode()).hexdigest()[:16]
                    submitted = ParseJob(doc_id=part_id, path=part.path,
                        out_dir=workspace/f'p{part.start_page:06d}')
                    expanded.append(submitted)
                    mapping[part_id] = (job.doc_id, version)
                    group.append((part, submitted))
                    existing = db.conn.execute('SELECT batch_id FROM ingest_parse_parts WHERE doc_id=? AND version=? AND part_id=?',
                                              (job.doc_id, version, part_id)).fetchone()
                    if existing and existing[0]:
                        prepared[part_id] = existing[0]
                part_groups[job.doc_id] = group
            else:
                expanded.append(job)
                mapping[job.doc_id] = (job.doc_id, db.get(job.doc_id)['version'])
                if job.doc_id in resume:
                    prepared[job.doc_id] = resume[job.doc_id]
        except Exception as exc:
            results[job.doc_id] = exc

    def on_batch(batch_id, submitted):
        for job in submitted:
            parent, version = mapping[job.doc_id]
            db.update(parent, status='parsing', mineru_batch_id=batch_id, error=None)
            if parent != job.doc_id:
                with db.transaction():
                    db.conn.execute('INSERT OR REPLACE INTO ingest_parse_parts VALUES (?,?,?,?)',
                                    (parent, version, job.doc_id, batch_id))
    if not expanded:
        return results
    async def execute(client):
        client.on_batch, client.resume_batches = on_batch, prepared
        return await client.parse_batch(expanded, poll_interval=cfg.ingest.mineru.poll_interval_s,
                                        timeout=cfg.ingest.mineru.timeout_s)
    if mineru is not None:
        parsed = await execute(mineru)
    else:
        async with MinerUClient(model_version=cfg.ingest.mineru.model_version, cfg=cfg.ingest.mineru) as client:
            parsed = await execute(client)
    for job in jobs:
        if job.doc_id in results:
            continue
        group = part_groups.get(job.doc_id)
        if group is None:
            results[job.doc_id] = parsed.get(job.doc_id, RuntimeError('MinerU未返回文件结果'))
            continue
        failure = next((parsed.get(j.doc_id, RuntimeError('MinerU缺少分段结果')) for _, j in group
                        if not isinstance(parsed.get(j.doc_id), Path)), None)
        if failure is not None:
            results[job.doc_id] = failure
        else:
            try:
                results[job.doc_id] = merge_parts([(part, parsed[j.doc_id]) for part, j in group],
                                                  job.out_dir, source_path=job.path)
            except Exception as exc:
                results[job.doc_id] = exc
    return results


def run_clean(input_dir=None, *, cfg=None, registry=None, mineru=None, limit=None, local_only=False,
              source_prefix="", source_names=None) -> dict:
    """完整扫描才标missing；失败重试不增版本，parsing+batch_id优先恢复。"""
    cfg = cfg or load_config()
    input_root = Path(input_dir if input_dir is not None else cfg.paths.raw).resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    if limit is not None and limit < 0:
        raise ValueError('limit必须>=0')
    prefix = source_prefix.strip("/")
    if prefix not in ("", "@uploads"):
        raise ValueError("非法登记前缀")
    def source_path(path):
        relative = path.relative_to(input_root).as_posix()
        return f"{prefix}/{relative}" if prefix else relative
    owned = registry is None
    db = registry or RegistryDB(cfg.paths.registry_db)
    stats = Counter(scanned=0, cleaned=0, skipped=0, duplicate=0, empty=0, unsupported=0, failed=0, pending=0, deleted=0)
    try:
        dedup = DocumentDedup(db, threshold=cfg.ingest.dedup.minhash_threshold)
        with db.transaction():
            db.conn.execute('CREATE TABLE IF NOT EXISTS ingest_sources(doc_id TEXT PRIMARY KEY,input_root TEXT NOT NULL)')
        excluded = [Path(getattr(cfg.paths, x)).resolve() for x in ('clean', 'parsed', 'chunks')]
        paths = sorted((p for p in input_root.rglob('*') if p.is_file() and not p.is_symlink() and
                p.resolve().is_relative_to(input_root) and not any(p.resolve().is_relative_to(d) for d in excluded)),
                key=lambda p: p.relative_to(input_root).as_posix())
        if local_only:
            paths = [p for p in paths if route(p, cfg=cfg.ingest.parse_routes) == 'text']
        present = {source_path(p) for p in paths}
        if source_names is not None:
            paths = [p for p in paths if p.relative_to(input_root).as_posix() in source_names]
        if limit is not None:
            paths = paths[:limit]
        cloud = []
        resume = {}
        for path in paths:
            stats['scanned'] += 1
            relative = source_path(path)
            doc_id = make_doc_id(relative)
            old_root = None
            try:
                parser = route(path, cfg=cfg.ingest.parse_routes)
                old_root = db.conn.execute('SELECT input_root FROM ingest_sources WHERE doc_id=?', (doc_id,)).fetchone()
                if old_root and old_root[0] != str(input_root):
                    raise ValueError('同一登记表包含另一个输入root下的相同相对路径')
                record, changed = db.register(relative, sha256(path), file_type=path.suffix.lstrip('.').lower(),
                    parser='mineru-'+cfg.ingest.mineru.model_version if parser == 'mineru' else 'text')
                with db.transaction():
                    db.conn.execute('INSERT OR REPLACE INTO ingest_sources VALUES (?,?)', (doc_id, str(input_root)))
                record['_input_root'] = str(input_root)
                artifact = Path(cfg.paths.clean)/doc_id/f"v{record['version']}"/'clean.json'
                if not changed and (record['status'] in {'empty', 'unsupported'} or
                        (record['status'] in {'cleaned', 'indexed'} and artifact.is_file())):
                    stats['skipped'] += 1
                    continue
                if parser is None:
                    db.update(doc_id, status='unsupported', error='不支持的文件格式')
                    stats['unsupported'] += 1
                    continue
                duplicate = db.exact_duplicate(record['content_hash'], exclude_doc_id=doc_id)
                if duplicate:
                    db.update(doc_id, status='duplicate', duplicate_of=duplicate['doc_id'], error=None)
                    stats['duplicate'] += 1
                    continue
                if parser == 'text':
                    status = _finish(path, record, cfg, db, dedup)
                    stats[status] += 1
                else:
                    from ragagent.ingest.mineru.client import ParseJob
                    cloud.append((ParseJob(doc_id=doc_id, path=path,
                        out_dir=Path(cfg.paths.parsed)/doc_id/f"v{record['version']}"), record))
                    if record['status'] == 'parsing' and record['mineru_batch_id']:
                        resume[doc_id] = record['mineru_batch_id']
            except Exception as exc:
                if db.get(doc_id) and not (old_root and old_root[0] != str(input_root)):
                    db.update(doc_id, status='failed', error=_safe_error(exc))
                log.warning('文件清洗失败 path=%s error_type=%s', relative, type(exc).__name__)
                stats['failed'] += 1
        if cloud:
            from ragagent.ingest.mineru.client import QuotaExceeded
            # 恢复排在新提交之前。每文件失败独立；没有token也不影响已完成的本地解析。
            cloud.sort(key=lambda pair: (pair[0].doc_id not in resume, pair[0].doc_id))
            try:
                results = asyncio.run(_cloud([j for j, _ in cloud], cfg, db, resume, mineru))
            except Exception as exc:
                results = {j.doc_id: exc for j, _ in cloud}
            for job, record in cloud:
                try:
                    result = results.get(job.doc_id, RuntimeError('MinerU没有返回文件结果'))
                    if isinstance(result, Exception):
                        raise result
                    duplicate = db.exact_duplicate(record['content_hash'], exclude_doc_id=job.doc_id)
                    if duplicate:
                        db.update(job.doc_id, status='duplicate', duplicate_of=duplicate['doc_id'], error=None)
                        stats['duplicate'] += 1
                    else:
                        stats[_finish(job.path, record, cfg, db, dedup, parsed_dir=result)] += 1
                except QuotaExceeded as exc:
                    db.update(job.doc_id, status='pending', error=_safe_error(exc))
                    stats['pending'] += 1
                except Exception as exc:
                    # timeout保留parsing/batch_id以便下次恢复，硬失败可重新提交。
                    current = db.get(job.doc_id)
                    status = 'parsing' if isinstance(exc, TimeoutError) and current['mineru_batch_id'] else 'failed'
                    db.update(job.doc_id, status=status, error=_safe_error(exc))
                    if status == 'failed':
                        with db.transaction():
                            db.conn.execute('DELETE FROM ingest_parse_parts WHERE doc_id=? AND version=?', (job.doc_id, record['version']))
                    stats['failed'] += 1
        if limit is None and not local_only and source_names is None:
            owned_ids = {r[0] for r in db.conn.execute('SELECT doc_id FROM ingest_sources WHERE input_root=?', (str(input_root),))}
            for record in db.list_documents():
                if record['doc_id'] in owned_ids and record['source_path'] not in present and record['status'] != 'deleted':
                    db.delete(record['doc_id'])
                    stats['deleted'] += 1
        return dict(stats)
    finally:
        if owned:
            db.close()


def main():
    parser = argparse.ArgumentParser(description='离线文档清洗')
    parser.add_argument('input_dir', nargs='?')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--local-only', action='store_true', help='只处理本地md/txt，不上传其他文件')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    # httpx/httpcore会在请求日志中记录presigned URL；CLI只保留警告和错误。
    for name in ('httpx', 'httpcore'):
        logging.getLogger(name).setLevel(logging.WARNING)
    from ragagent.ingest.lock import IngestBusy, ingest_lock
    cfg = load_config(args.config)
    try:
        with ingest_lock(cfg.paths.ingest_lock):
            report = run_clean(args.input_dir, cfg=cfg, limit=args.limit, local_only=args.local_only)
    except IngestBusy as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
