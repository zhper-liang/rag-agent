"""错误分类与无敏感正文的异常描述。"""
import openai


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, openai.APITimeoutError)):
        return False
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


def error_meta(exc: BaseException) -> dict:
    out = {'error': type(exc).__name__}
    status = getattr(exc, 'status_code', None)
    if isinstance(status, int):
        out['status_code'] = status
    return out


class FirstTokenError(Exception):
    """尚未输出可见正文，可重试；cause 只内部使用，不记录 repr。"""
    def __init__(self, cause: BaseException, fatal: bool):
        super().__init__(type(cause).__name__)
        self.cause, self.fatal = cause, fatal


class GenerationFailed(Exception):
    def __init__(self, reason='all endpoints failed', attempts=None):
        super().__init__(reason)
        self.attempts = attempts or []
