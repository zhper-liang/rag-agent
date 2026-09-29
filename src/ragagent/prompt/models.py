"""模型无关的消息结构：Part、Message、PromptPackage（Prompt 组装.md §10）。"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class TextPart(BaseModel):
    text: str
    type: Literal["text"] = "text"


class ImagePart(BaseModel):
    media_type: str            # image/jpeg
    data_b64: str
    chunk_id: str
    type: Literal["image"] = "image"


Part = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    parts: list[Part]


class PromptPackage(BaseModel):
    scene: Literal["rag", "rag_empty", "chat"]
    system: str
    messages: list[Message]
    version: str               # prompt.version，记进日志
    est_input_tokens: int
    meta: dict = Field(default_factory=dict)
