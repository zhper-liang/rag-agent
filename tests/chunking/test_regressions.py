

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


def test_merged_table_page_end_and_image_context_survive_chunking():
    from ragagent.ingest.models import CleanDocument, Block
    from tests.index.helpers import config
    from ragagent.ingest.chunking.pipeline import run_chunk
    doc=CleanDocument(doc_id='abcdef0123456789',version=1,source_path='table.pdf',file_type='pdf',parser='mineru-vlm',
        title='表格',content_hash='a'*64,blocks=[Block(type='table',text='|A|B|\n|---|---|\n|1|2|',page=1,meta={'page_end':3}),
        Block(type='image',text='',image_path='clean/abcdef0123456789/v1/images/a.png',page=3,meta={'context':'来自清洗器的前后文'})])
    result=run_chunk(doc,cfg=config().chunking,count=len)
    text=[c for c in result if c.type=='text']
    assert text[0].page_start==1 and text[0].page_end==3
    assert [c for c in result if c.type=='image'][0].text=='来自清洗器的前后文'
