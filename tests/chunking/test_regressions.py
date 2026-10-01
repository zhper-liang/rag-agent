

def test_default_semantic_encoder_is_lazily_wired(monkeypatch):
    from ragagent.ingest.chunking import pipeline
    from tests.chunking.test_properties import config, document
    from ragagent.ingest.models import Block
    cfg = config()
    cfg = cfg.model_copy(update={'semantic': cfg.semantic.model_copy(update={
        'enabled': True, 'min_units': 4})})
    calls = []
    def encode(texts):
        calls.append(texts)
        return [[1., 0.] if i % 3 else [0., 1.] for i, _ in enumerate(texts)]
    monkeypatch.setattr(pipeline, '_service_encode', encode)
    doc = document([Block(type='paragraph', text='部署说明。' * 10 + str(i)) for i in range(8)])
    first = pipeline.run_chunk(doc, cfg=cfg, count=len)
    assert calls and all(c.tokens <= cfg.max_tokens for c in first)
    assert [c.model_dump() for c in first] == [c.model_dump() for c in
        pipeline.run_chunk(doc, cfg=cfg, count=len, encode=encode)]
