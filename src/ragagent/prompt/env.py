"""可信模板与不可信变量的边界（Prompt 组装 §5）。"""
from __future__ import annotations

import html
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

RESERVED = r"documents|document|question|standalone|scope|history|hints|tried"
_TAG = re.compile(rf"<(/?)\s*({RESERVED})\b", re.I)
DEFAULT_PROMPTS = Path(__file__).resolve().parents[3] / "prompts" / "v1"


def escape_tags(text: str) -> str:
    """只替换保留标签的左尖括号；不 HTML 转义正文、代码。"""
    return _TAG.sub(lambda m: f"＜{m.group(1)}{m.group(2)}", text)


def attr(value: str) -> str:
    return html.escape(value, quote=True)


def build_env(path: str | Path = DEFAULT_PROMPTS) -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(path)), autoescape=False, undefined=StrictUndefined
    )
    env.filters.update(esc=escape_tags, attr=attr)
    return env


env = build_env()
