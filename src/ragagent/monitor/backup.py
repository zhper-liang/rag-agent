"""SQLite 在线备份（包含已提交 WAL），只轮转已成功发布的快照。"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from ragagent.ingest.lock import ingest_lock


def backup_databases(cfg):
    directory = Path(cfg.backup.dir)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    report = {'backups': {}, 'skipped': [], 'errors': {}}
    with ingest_lock(directory / '.backup.lock'):
        for key in cfg.backup.dbs:
            source = Path(getattr(cfg.paths, key)).resolve()
            if not source.is_file():
                report['skipped'].append(key)
                continue
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
            target = directory / f'{key}-{stamp}.sqlite3'
            fd, name = tempfile.mkstemp(prefix=f'.{key}-', suffix='.partial', dir=directory)
            os.close(fd)
            partial = Path(name)
            try:
                src = sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)
                try:
                    dst = sqlite3.connect(partial)
                    try:
                        src.backup(dst)
                        if dst.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                            raise RuntimeError('backup integrity check failed')
                    finally:
                        dst.close()
                finally:
                    src.close()
                with partial.open('rb') as handle:
                    os.fsync(handle.fileno())
                partial.replace(target)
                report['backups'][key] = str(target)
                for old in sorted(directory.glob(f'{key}-*.sqlite3'), reverse=True)[cfg.backup.keep:]:
                    old.unlink()
            except Exception as exc:
                report['errors'][key] = type(exc).__name__
            finally:
                partial.unlink(missing_ok=True)
    return report


def main(argv=None):
    import argparse
    from ragagent.config import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args(argv)
    report = backup_databases(load_config(args.config))
    print(json.dumps(report, ensure_ascii=False))
    return 1 if report['errors'] else 0
