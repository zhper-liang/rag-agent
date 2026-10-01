"""编码/重排使用同一 HTTP 协议；client 可注入，默认按配置创建短生命周期客户端。"""

import httpx


async def post_json(client, cfg, path: str, payload: dict, timeout_s: float):
    if client is None:
        address = f"http://{cfg.services.embed.host}:{cfg.services.embed.port}"
        async with httpx.AsyncClient(base_url=address, timeout=timeout_s) as owned:
            response = await owned.post(path, json=payload)
    else:
        response = await client.post(path, json=payload, timeout=timeout_s)
    response.raise_for_status()
    return response.json()
