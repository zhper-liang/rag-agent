import numpy as np
import pytest
from PIL import Image
from ragagent.config import load_config
from ragagent.ingest.models import Block
from ragagent.ingest.images import process_images, prune_images
from ragagent.ingest.filters import filter_quality, garbled_ratio, is_empty
from ragagent.ingest.dedup import DocumentDedup, minhash, sha256
from ragagent.ingest.registry_db import RegistryDB


def textured(path,seed=1,size=128):
    Image.fromarray(np.random.default_rng(seed).integers(0,256,(size,size,3),dtype=np.uint8)).save(path)


def test_image_filter_context_duplicates_and_original(tmp_path):
    (tmp_path/'images').mkdir()
    textured(tmp_path/'images'/'large.png')
    textured(tmp_path/'images'/'small.png',size=32)
    Image.new('RGB',(128,128),'red').save(tmp_path/'images'/'blank.png')
    blocks=[Block(type='paragraph',text='前面段落'*100),Block(type='image',text='',image_path='images/large.png'),
            Block(type='paragraph',text='后面段落'*100),Block(type='image',text='重复',image_path='images/large.png'),
            Block(type='image',text='',image_path='images/small.png'),Block(type='image',text='',image_path='images/blank.png')]
    result=process_images(blocks,tmp_path)
    images=[b for b in result if b.type=='image']
    assert len(images)==1 and images[0].meta['context']=='前面段落'*50+'\n'+'后面段落'*50
    assert 'phash' in images[0].meta and 'context' not in blocks[1].meta
    prune_images(result,tmp_path)
    assert [p.name for p in (tmp_path/'images').glob('*')]==['large.png']


def test_logo_page_ratio_and_caption(tmp_path):
    (tmp_path/'images').mkdir()
    textured(tmp_path/'images'/'logo.png')
    textured(tmp_path/'images'/'figure.png',seed=2)
    blocks=[Block(type='paragraph',text='正文',page=p) for p in range(1,5)]
    blocks += [Block(type='image',text='logo',page=p,image_path='images/logo.png') for p in (1,2)]
    blocks += [Block(type='image',text='图说明',page=3,image_path='images/figure.png')]
    images=[b for b in process_images(blocks,tmp_path) if b.type=='image']
    assert len(images)==1 and images[0].meta['context']=='图说明'
    assert len(process_images([blocks[-1]],tmp_path))==1  # 单页不误删正文图。


def test_llm_description_not_silently_ignored(tmp_path):
    cfg=load_config().ingest.images.model_copy(update={'llm_description':True})
    with pytest.raises(NotImplementedError):
        process_images([],tmp_path,cfg=cfg)


def test_quality_and_document_empty():
    blocks=[Block(type='paragraph',text='短'),Block(type='heading',text='短',level=1),Block(type='image',text=''),
            Block(type='paragraph',text='\ufffd'*20),Block(type='code',text='```\nx += 1\n```'),Block(type='paragraph',text='有效的中文与English123，标点。')]
    assert [b.type for b in filter_quality(blocks)]==['heading','image','code','paragraph']
    assert not is_empty([Block(type='image',text='')]) and is_empty([])
    assert is_empty([Block(type='paragraph',text='有效的简短文字')])
    assert garbled_ratio('abc中文，123')==0 and garbled_ratio('\ufffd'*5)==1


def test_persistent_minhash_exact_and_current_version(tmp_path):
    path=tmp_path/'source.txt'; path.write_text('内容')
    assert len(sha256(path))==64
    text='部署与维护手册。'+''.join(f'具体步骤{i}配置安装成功。' for i in range(80))
    assert minhash(text).jaccard(minhash(text+'增加一处注释'))>.9
    assert minhash('') is None
    with RegistryDB(tmp_path/'r.db') as db:
        record,_=db.register('a.md','hash')
        db.set_status(record['doc_id'],'cleaned')
        dedup=DocumentDedup(db)
        dedup.remember(record['doc_id'],1,text)
        assert dedup.duplicate_of(text+'增加一处注释')==record['doc_id']
        assert dedup.duplicate_of(text,exclude_doc_id=record['doc_id']) is None
        assert dedup.duplicate_of('完全不同的文本'*10) is None
        db.register('a.md','changed')
        db.set_status(record['doc_id'],'cleaned')
        assert dedup.duplicate_of(text) is None  # 旧version签名不能匹配。
        dedup.remember(record['doc_id'],2,text)
    with RegistryDB(tmp_path/'r.db') as db:
        assert DocumentDedup(db).duplicate_of(text)==record['doc_id']
        db.delete(record['doc_id'])
        assert DocumentDedup(db).duplicate_of(text) is None


def test_image_hash_and_copy_use_same_source_snapshot(tmp_path, monkeypatch):
    import hashlib
    from ragagent.ingest.images import copy_image
    source = tmp_path/'source.png'
    before, after = b'bytes-at-read-time', b'bytes-after-concurrent-replacement'
    source.write_bytes(before)
    read_bytes = type(source).read_bytes
    def changing(path):
        value = read_bytes(path)
        if path == source:
            path.write_bytes(after)
        return value
    monkeypatch.setattr(type(source), 'read_bytes', changing)
    copied = tmp_path/'out'/copy_image(source, tmp_path/'out')
    assert copied.read_bytes() == before
    assert copied.stem == hashlib.sha256(copied.read_bytes()).hexdigest()
