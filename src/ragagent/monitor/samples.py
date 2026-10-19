"""完整 Prompt/回答只写本地；绝不进入 trace、state 或 checkpoint。"""
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import random
import threading
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class PrivateRotatingHandler(RotatingFileHandler):
    def _open(self):
        fd = os.open(self.baseFilename, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.fchmod(fd, 0o600)
        return os.fdopen(fd, self.mode, encoding=self.encoding, errors=self.errors)

    def handleError(self, record):
        # logging 的默认 stderr 回退会打印 record 内容；禁止原文逃逸到 journald。
        raise OSError('local prompt sample write failed')


class PromptSampler:
    def __init__(self, cfg):
        self.settings = cfg.logging
        self.config_hash = hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()
        self.handler = None
        self.lock = threading.Lock()

    def record(self, package, answer, meta):
        if self.settings.prompt_sample_rate <= 0 or random.random() >= self.settings.prompt_sample_rate:
            return False
        try:
            payload = {'timestamp': datetime.now(timezone.utc).isoformat(),
                       'config_hash': self.config_hash, 'prompt': package.model_dump(),
                       'answer': answer, 'status': meta['status']}
            line = json.dumps(payload, ensure_ascii=False)
            with self.lock:
                if self.handler is None:
                    path = Path(self.settings.sample_path)
                    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    self.handler = PrivateRotatingHandler(path, maxBytes=self.settings.sample_max_mb * 1024 ** 2,
                        backupCount=self.settings.sample_keep_files, encoding='utf-8')
                self.handler.emit(logging.LogRecord('local_prompt_sample', logging.INFO, '', 0, line, (), None))
            return True
        except Exception as exc:
            log.warning('local prompt sample failed_type=%s', type(exc).__name__)
            return False

    def close(self):
        with self.lock:
            if self.handler is not None:
                self.handler.close()
                self.handler = None
