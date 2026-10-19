"""CLI 级入库互斥；文件保留，不能 unlink 以免竞争进程锁住不同 inode。"""
from contextlib import contextmanager
import fcntl
from pathlib import Path


class IngestBusy(RuntimeError):
    """已有离线写入进程。"""


@contextmanager
def ingest_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise IngestBusy('已有入库任务运行，未执行任何写入') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
