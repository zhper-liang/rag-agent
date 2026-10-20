"""OpenAI 兼容接口的桩（总揽.md §4.4）。

- POST /v1/chat/completions：流式（SSE）与非流式；带 tools 时返回 tool call（结构化输出）
- 回答正文取自 tests/fixtures/answers/{name}.md，tool call 参数取自对应的夹具
- 故障注入：POST /_stub/script 按模型排队若干步，每次请求取一步：
  status=429/5xx 直接返回错误；delay_s 在首个正文 chunk（非流式为整个响应）之前等待，模拟首字超时
- GET /_stub/calls 查看收到的请求（模型名、是否流式、tool 名），POST /_stub/reset 清空
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def _load(name: str, index: int | None = None) -> dict:
    data = json.loads((FIX / name).read_text(encoding="utf-8"))
    return data[index] if index is not None else data


# 结构化输出的 tool 名（pydantic 类名）→ 默认参数
TOOL_ARGS = {
    "Preprocessed": lambda: _load("preprocess_out.json", 0),
    "Rewrite": lambda: _load("rewrite_out.json"),
}


class Step(BaseModel):
    status: int | None = None          # 注入的 HTTP 错误码
    delay_s: float = 0.0
    answer: str = Field("basic", pattern=r"^[a-z_]+$")   # answers/ 下的文件名
    chunk_chars: int = Field(6, ge=1)  # 每个流式 chunk 的字符数，引用会被切在边界上
    tool_args: dict | str | None = None   # 覆盖 tool call 参数；传字符串可模拟非法 JSON


class Script(BaseModel):
    model: str = "*"                   # 只对这个模型生效；"*" 为所有模型
    steps: list[Step]


def _error(status: int) -> JSONResponse:
    kind = "rate_limit_exceeded" if status == 429 else "server_error"
    return JSONResponse({"error": {"message": f"stub injected {status}", "type": kind, "code": kind}},
                        status_code=status)


def _usage(body: dict, completion: str) -> dict:
    prompt = sum(len(json.dumps(m.get("content"), ensure_ascii=False)) for m in body.get("messages", []))
    return {"prompt_tokens": prompt, "completion_tokens": len(completion),
            "total_tokens": prompt + len(completion)}


def create_app() -> FastAPI:
    app = FastAPI(title="llm-stub")
    queues: dict[str, deque[Step]] = defaultdict(deque)
    calls: list[dict] = []

    def next_step(model: str) -> Step:
        for key in (model, "*"):
            if queues[key]:
                return queues[key].popleft()
        return Step()

    @app.post("/_stub/script")
    def script(s: Script) -> dict:
        queues[s.model].extend(s.steps)
        return {"queued": len(queues[s.model])}

    @app.post("/_stub/reset")
    def reset() -> dict:
        queues.clear()
        calls.clear()
        return {}

    @app.get("/_stub/calls")
    def get_calls() -> list[dict]:
        return calls

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        model, stream = body.get("model", ""), bool(body.get("stream"))
        tool = _tool_name(body)
        step = next_step(model)
        calls.append({"model": model, "stream": stream, "tool": tool, "status": step.status})
        if step.status:
            return _error(step.status)

        cid, created = f"chatcmpl-stub-{len(calls)}", int(time.time())
        if tool:
            if tool not in TOOL_ARGS:
                return JSONResponse({"error": {"message": f"unknown tool {tool}", "type": "invalid_request_error"}},
                                    status_code=400)
            args = step.tool_args if step.tool_args is not None else TOOL_ARGS[tool]()
            tool_call = {"id": f"call_{len(calls)}", "type": "function",
                         "function": {"name": tool, "arguments": args if isinstance(args, str)
                                      else json.dumps(args, ensure_ascii=False)}}
            content, message = "", {"role": "assistant", "content": None, "tool_calls": [tool_call]}
        else:
            content = (FIX / "answers" / f"{step.answer}.md").read_text(encoding="utf-8")
            message = {"role": "assistant", "content": content}
        finish = "tool_calls" if tool else "stop"
        usage = _usage(body, content)

        if not stream:
            await asyncio.sleep(step.delay_s)
            return {"id": cid, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}

        def chunk(delta: dict, finish_reason=None, **extra) -> str:
            data = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}], **extra}
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        async def events():
            yield chunk({"role": "assistant", "content": ""})     # 响应头和首个空 chunk 立即返回
            await asyncio.sleep(step.delay_s)                    # 首字超时发生在正文之前
            if tool:
                yield chunk({"tool_calls": [{"index": 0, **tool_call}]})
            else:
                for i in range(0, len(content), step.chunk_chars):
                    yield chunk({"content": content[i:i + step.chunk_chars]})
            yield chunk({}, finish)
            if (body.get("stream_options") or {}).get("include_usage"):
                data = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [], "usage": usage}
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def _tool_name(body: dict) -> str | None:
    """强制调用的 tool 名；没有 tools 时返回 None。"""
    if not body.get("tools"):
        return None
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        return choice["function"]["name"]
    return body["tools"][0]["function"]["name"]


if __name__ == "__main__":
    from tests.stubs.serve import main
    main(create_app(), 8003)
