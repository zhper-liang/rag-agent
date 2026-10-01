"""一次请求的中间消息包；原图只驻留 generate 内部，不写 checkpoint。"""
from __future__ import annotations

import logging
import math
import re
import secrets
import unicodedata
from pathlib import Path

from ragagent.context.models import ContextBlock
from ragagent.prompt.env import build_env, escape_tags
from ragagent.prompt.history import history_turns, trim_history
from ragagent.prompt.images import load_image
from ragagent.prompt.models import ImagePart, Message, PromptPackage, TextPart

log = logging.getLogger(__name__)


def _key(text: str) -> str:
    return ''.join(unicodedata.normalize('NFKC', text).lower().split())


def build_user_rag(env, blocks, question, standalone, scope_note, load_image, *, max_images=2,
                   meta=None) -> Message:
    nonce, images = secrets.token_hex(16), []
    meta = {} if meta is None else meta

    def image_marker(image):
        if len(images) >= max_images:
            return ''
        images.append(image)
        return f'@@IMG:{nonce}:{len(images) - 1}@@'

    text = env.get_template('user_rag.j2').render(blocks=blocks, question=question,
                                               standalone=standalone, scope_note=scope_note,
                                               image_marker=image_marker)
    parts, failed = [], 0
    for i, segment in enumerate(re.split(rf'@@IMG:{nonce}:(\d+)@@', text)):
        if i % 2 == 0:
            if segment:
                parts.append(TextPart(text=segment))
        else:
            image = load_image(images[int(segment)])
            if image is None:
                failed += 1
            else:
                parts.append(image)
    meta['images_failed'] = failed
    return Message(role='user', parts=parts)


def assemble(state, cfg, *, count, env=None, image_loader=None) -> PromptPackage:
    env = env if env is not None else build_env(Path(cfg.prompt.dir) / cfg.prompt.version)
    blocks = [ContextBlock.model_validate(b) for b in state.get('context_blocks', [])]
    intent, question = state.get('intent', 'rag'), state['question']
    meta = {}
    if intent == 'chat':
        scene = 'chat'
        system = env.get_template('system_chat.j2').render()
        current = Message(role='user', parts=[TextPart(text=escape_tags(question))])
    elif intent == 'rag':
        scene = 'rag' if blocks else 'rag_empty'
        system = env.get_template('system_rag.j2').render(supplement_heading=cfg.prompt.supplement_heading)
        standalone = state.get('standalone_question', '')
        if _key(standalone) == _key(question):
            standalone = ''
        if blocks:
            current = build_user_rag(env, blocks, question, standalone, state.get('scope_note'),
                                     image_loader or (lambda image: load_image(image, cfg)),
                                     max_images=cfg.context.image.max_images, meta=meta)
        else:
            text = env.get_template('user_rag_empty.j2').render(question=question, standalone=standalone,
                                                              scope_note=state.get('scope_note'))
            current = Message(role='user', parts=[TextPart(text=text)])
    else:
        raise ValueError('clarify/fallback 不调用 assemble')
    turns = history_turns(state.get('messages', []))
    history = [Message(role=m['role'], parts=[TextPart(text=escape_tags(m['content']))])
               for m in trim_history(turns, cfg.prompt, count)]

    def text_tokens():
        return math.ceil((count(system) + sum(count(p.text) for m in [*history, current]
                                             for p in m.parts if isinstance(p, TextPart)))
                         * cfg.context.token_safety_factor)

    cost = text_tokens()
    while history and cost > cfg.prompt.max_input_tokens:
        del history[:2]  # 只删最老完整轮次；资料永远不改
        cost = text_tokens()
    meta['history_dropped_turns'] = len(turns) - len(history) // 2
    meta['text_tokens'] = cost
    meta['over_budget'] = cost > cfg.prompt.max_input_tokens
    if meta['over_budget']:
        log.warning('prompt exceeds text token budget after dropping history')
    image_tokens = sum(isinstance(p, ImagePart) for p in current.parts) * cfg.context.image.tokens_per_image
    meta['image_tokens'] = image_tokens
    return PromptPackage(scene=scene, system=system, messages=[*history, current], version=cfg.prompt.version,
                         est_input_tokens=cost + image_tokens, meta=meta)
