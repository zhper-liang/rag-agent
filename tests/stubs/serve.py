"""在后台线程里起 uvicorn，供测试使用；桩只监听本机。"""

import argparse
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import uvicorn
from fastapi import FastAPI

HOST = "127.0.0.1"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


@contextmanager
def serve(app: FastAPI, port: int = 0) -> Iterator[str]:
    """启动 app，产出 base URL；退出时关闭。port=0 时自动选空闲端口。"""
    port = port or _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=port, log_level="warning",
                                           timeout_graceful_shutdown=1))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if not t.is_alive() or time.monotonic() > deadline:
            raise RuntimeError(f"桩启动失败：{HOST}:{port}")
        time.sleep(0.01)
    try:
        yield f"http://{HOST}:{port}"
    finally:
        server.should_exit = True
        t.join(5)


def main(app: FastAPI, default_port: int) -> None:
    """命令行入口：python -m tests.stubs.xxx_server [--port N]"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=default_port)
    uvicorn.run(app, host=HOST, port=ap.parse_args().port)
