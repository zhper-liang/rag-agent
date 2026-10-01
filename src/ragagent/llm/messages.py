"""PromptPackage → LangChain 消息；按当前接入点降级图片。"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def to_messages(pkg, vision: bool) -> tuple[list, int]:
    out, dropped = [SystemMessage(content=pkg.system)], 0
    for message in pkg.messages:
        blocks = []
        for part in message.parts:
            if part.type == 'text':
                blocks.append({'type': 'text', 'text': part.text})
            elif vision:
                blocks.append({'type': 'image_url', 'image_url': {
                    'url': f'data:{part.media_type};base64,{part.data_b64}'}})
            else:
                dropped += 1
        content = blocks if any(p['type'] == 'image_url' for p in blocks) else ''.join(p['text'] for p in blocks)
        cls = HumanMessage if message.role == 'user' else AIMessage
        out.append(cls(content=content))
    return out, dropped
