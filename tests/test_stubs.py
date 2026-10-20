"""三个桩的冒烟测试：走真实 HTTP，LLM 桩用真实的 ChatOpenAI 客户端调用（总揽.md §4.4）。"""

import asyncio
import json
import math
from pathlib import Path

import httpx
import openai
import pytest
from langchain_openai import ChatOpenAI

from ragagent.index.models import EmbedResponse, RerankResponse
from ragagent.preprocess.schema import Preprocessed, Rewrite

FIX = Path(__file__).resolve().parent / "fixtures"


# ---- 编码

def test_embed_text(embed_url):
    r = httpx.post(f"{embed_url}/embed/text", json={"texts": ["milvus 配置", "milvus 配置", "别的"]})
    vecs = EmbedResponse.model_validate(r.json()).vectors
    assert [len(v) for v in vecs] == [1024] * 3
    assert all(math.isclose(math.fsum(x * x for x in v), 1.0) for v in vecs)
    assert vecs[0] == vecs[1] != vecs[2]


def test_embed_image(embed_url):
    ok = httpx.post(f"{embed_url}/embed/image", json={"items": [{"image_path": "a/fig.jpg", "text": "图"}]})
    assert len(ok.json()["vectors"][0]) == 1024
    bad = httpx.post(f"{embed_url}/embed/image", json={"items": [{"image_path": "../etc/passwd", "text": ""}]})
    assert bad.status_code == 422


def test_embed_health(embed_url):
    assert httpx.get(f"{embed_url}/health").json()["dim"] == 1024


# ---- 重排

def test_rerank(embed_url):
    passages = ["把 queryNode.enableDisk 设为 true", "Grafana 导入官方面板", "字" * 50 + "enableDisk"]
    r = httpx.post(f"{embed_url}/rerank", json={"query": "enableDisk 怎么设", "passages": passages,
                                                "max_length": 20})
    s = RerankResponse.model_validate(r.json()).scores
    assert all(0 <= x <= 1 for x in s)
    assert s[0] > s[1]
    assert s[2] < s[0]                        # 匹配词在截断位置之后，不计分


def test_rerank_pairs(embed_url):
    r = httpx.post(f"{embed_url}/rerank/pairs", json={"pairs": [["备份", "备份前先停止写入"], ["备份", "升级"]],
                                                      "max_length": 512})
    a, b = r.json()["scores"]
    assert a == 1.0 and b == 0.0


# ---- LLM

def _client(url, model="m-main"):
    return ChatOpenAI(base_url=f"{url}/v1", api_key="stub", model=model, max_retries=0,
                      stream_usage=True, timeout=10)


def _script(url, *steps, model="*"):
    httpx.post(f"{url}/_stub/script", json={"model": model, "steps": list(steps)}).raise_for_status()


def test_llm_stream(llm_url):
    async def run():
        chunks, usage = [], None
        async for c in _client(llm_url).astream("hi"):
            chunks.append(c.content)
            usage = c.usage_metadata or usage
        return chunks, usage

    chunks, usage = asyncio.run(run())
    assert "".join(chunks) == (FIX / "answers" / "basic.md").read_text(encoding="utf-8")
    assert sum(bool(c) for c in chunks) > 10 and usage["output_tokens"] > 0


def test_llm_invoke_answer(llm_url):
    _script(llm_url, {"answer": "supplement"})
    msg = _client(llm_url).invoke("hi")
    assert msg.content == (FIX / "answers" / "supplement.md").read_text(encoding="utf-8")


def test_llm_structured(llm_url):
    pre = _client(llm_url).with_structured_output(Preprocessed, method="function_calling").invoke("q")
    expected = json.loads((FIX / "preprocess_out.json").read_text(encoding="utf-8"))[0]
    assert pre.model_dump() == expected
    rw = _client(llm_url).with_structured_output(Rewrite, method="function_calling").invoke("q")
    assert rw.queries
    assert [c["tool"] for c in httpx.get(f"{llm_url}/_stub/calls").json()] == ["Preprocessed", "Rewrite"]


def test_llm_structured_bad_output(llm_url):
    # langchain-openai 1.x：参数不是合法 JSON 时不抛异常而是返回 None；字段不符才抛 ValidationError。
    # structured_call（大模型生成.md §6.2）必须把 None 当失败处理
    _script(llm_url, {"tool_args": "{bad"}, {"tool_args": {"queries": "not-a-list"}})
    llm = _client(llm_url).with_structured_output(Rewrite, method="function_calling")
    assert llm.invoke("q") is None
    with pytest.raises(ValueError):
        llm.invoke("q")


@pytest.mark.parametrize(("status", "exc"), [(429, openai.RateLimitError), (500, openai.InternalServerError),
                                             (503, openai.InternalServerError)])
def test_llm_injected_errors(llm_url, status, exc):
    _script(llm_url, {"status": status}, model="m-main")
    with pytest.raises(exc):
        _client(llm_url).invoke("hi")
    assert _client(llm_url, "m-backup").invoke("hi").content    # 只对指定模型注入
    assert _client(llm_url).invoke("hi").content                # 每步只用一次


def test_llm_first_token_timeout(llm_url):
    _script(llm_url, {"delay_s": 2})

    async def first_content(timeout):
        it = _client(llm_url).astream("hi").__aiter__()
        try:
            while True:
                c = await asyncio.wait_for(it.__anext__(), timeout)
                if c.content:
                    return c.content
        finally:
            await it.aclose()

    with pytest.raises(TimeoutError):
        asyncio.run(first_content(0.3))
    assert asyncio.run(first_content(5))                         # 下一次请求不再延迟
