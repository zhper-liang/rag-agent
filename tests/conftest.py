import httpx
import pytest

from tests.stubs import embed_server, llm_server
from tests.stubs.serve import serve


@pytest.fixture(scope="session")
def embed_url():
    """编码 + 重排桩（同一进程，与真实服务一致）。"""
    with serve(embed_server.create_app()) as url:
        yield url


@pytest.fixture(scope="session")
def _llm_server():
    with serve(llm_server.create_app()) as url:
        yield url


@pytest.fixture
def llm_url(_llm_server):
    """LLM 桩；每个用例开始前清空脚本和调用记录。"""
    httpx.post(f"{_llm_server}/_stub/reset").raise_for_status()
    return _llm_server
