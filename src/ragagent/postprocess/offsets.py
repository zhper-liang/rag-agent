"""前端 JavaScript 索引是 UTF-16 单元，不是 Unicode code points。"""
def utf16_offset(text: str, position: int) -> int:
    return len(text[:position].encode('utf-16-le', errors='surrogatepass')) // 2


def utf16_citations(answer: str, sents) -> list[dict]:
    return [dict(c, start=utf16_offset(answer, c['start']), end=utf16_offset(answer, c['end']))
            for s in sents for c in s.cites]
