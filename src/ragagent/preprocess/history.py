"""只把最近完整问答的纯文字带入预处理，不携带工具消息或图片。"""


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def history_turns(messages) -> list[dict[str, str]]:
    turns, question = [], None
    for message in messages or []:
        if isinstance(message, dict):
            role, content = message.get("role", message.get("type")), message.get("content", "")
            plain = message.get("additional_kwargs", {}).get("plain", message.get("plain"))
        else:
            role, content = getattr(message, "type", None), getattr(message, "content", "")
            plain = getattr(message, "additional_kwargs", {}).get("plain")
        if role in {"user", "human"}:
            question = _text(content)
        elif role in {"assistant", "ai"} and question is not None:
            turns.append({"question": question, "answer": plain if isinstance(plain, str) else _text(content)})
            question = None
    return turns


def truncate_text(text: str, tokens: int, count) -> str:
    if tokens <= 0:
        return ""
    if count(text) <= tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(text[:mid]) <= tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def history_text(turns) -> str:
    return "\n".join(f"用户：{t['question']}\n助手：{t['answer']}" for t in turns)


def truncate_history(messages, cfg, count) -> list[dict[str, str]]:
    if cfg.turns <= 0 or cfg.total_tokens <= 0:
        return []
    turns = [{"question": t["question"], "answer": truncate_text(t["answer"], cfg.answer_tokens, count)}
             for t in history_turns(messages)[-cfg.turns:]]
    while len(turns) > 1 and count(history_text(turns)) > cfg.total_tokens:
        turns.pop(0)
    if turns and count(history_text(turns)) > cfg.total_tokens:
        # 保留最新用户问题优先；计数包括角色标识、换行，而非只数正文。
        t = turns[0]
        lo, hi = 0, len(t["question"])
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count(history_text([{"question": t["question"][:mid], "answer": ""}])) <= cfg.total_tokens:
                lo = mid
            else:
                hi = mid - 1
        t["question"] = t["question"][:lo]
        lo, hi = 0, len(t["answer"])
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count(history_text([{"question": t["question"], "answer": t["answer"][:mid]}])) <= cfg.total_tokens:
                lo = mid
            else:
                hi = mid - 1
        t["answer"] = t["answer"][:lo]
        if count(history_text(turns)) > cfg.total_tokens:
            return []
    return turns


def previous_intent(messages):
    for message in reversed(messages or []):
        if isinstance(message, dict):
            role = message.get("role", message.get("type"))
            extra = message.get("additional_kwargs", {})
            intent = extra.get("intent", message.get("intent"))
        else:
            role = getattr(message, "type", None)
            intent = getattr(message, "additional_kwargs", {}).get("intent")
        if role in {"assistant", "ai"}:
            return intent if intent in {"rag", "chat", "clarify"} else None
    return None
