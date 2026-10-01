"""仅规则处理，不调用外部服务（问题预处理 §3）。"""

import unicodedata


class QuestionError(ValueError):
    """可直接展示的输入提示，不包含原始问题。"""


def normalize_question(question: str, max_chars: int) -> str:
    if not isinstance(question, str):
        raise QuestionError("请输入有效的问题")
    text = unicodedata.normalize("NFKC", question)
    text = "".join(c for c in text if unicodedata.category(c) not in {"Cc", "Cf", "Cs"}).strip()
    if not text:
        raise QuestionError("请输入问题")
    if len(text) > max_chars:
        raise QuestionError("问题过长，请精简后再问")
    return text
