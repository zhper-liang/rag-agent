"""清洗产物：Block 与 CleanDocument（数据清洗.md §6）。"""

from typing import Literal

from pydantic import BaseModel, Field

BlockType = Literal["heading", "paragraph", "table", "list",
                    "code", "equation", "image"]

DOC_ID_PATTERN = r"^[0-9a-f]{16}$"


class Block(BaseModel):
    type: BlockType
    text: str                      # table 用 Markdown；equation 用 LaTeX；image 用说明文字
    level: int = 0                 # heading 的层级 1~6
    page: int | None = None        # 页码（1 起），PPT 为幻灯片序号
    image_path: str | None = None  # image / table 截图的相对路径
    meta: dict = Field(default_factory=dict)


class CleanDocument(BaseModel):
    doc_id: str = Field(pattern=DOC_ID_PATTERN)  # sha1(相对路径)[:16]，版本之间保持不变
    version: int
    source_path: str
    file_type: str       # pdf / docx / pptx / png / md / ...
    parser: str          # mineru-vlm / mineru-pipeline / text
    title: str
    content_hash: str    # 原文件 sha256
    blocks: list[Block]
    meta: dict = Field(default_factory=dict)
