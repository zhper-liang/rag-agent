import json
from pathlib import Path
import pytest
from ragagent.config import load_config
from ragagent.ingest.pipeline import run_clean, render_markdown
from ragagent.ingest.registry_db import RegistryDB, make_doc_id
from ragagent.ingest.models import Block, CleanDocument
from ragagent.ingest.mineru.client import QuotaExceeded


def config(tmp_path):
    cfg=load_config()
    paths=cfg.paths.model_copy(update={k:str(tmp_path/'data'/k) for k in ('raw','parsed','clean','chunks')})
    paths=paths.model_copy(update={'registry_db':str(tmp_path/'data/registry.db')})
    return cfg.model_copy(update={'paths':paths})


def article(topic):
    return f'# {topic}\n\n'+''.join(f'{topic}第{i}条操作说明，这是完整且有效的正文。\n' for i in range(30))


def test_incremental_duplicate_empty_unsupported_fail_and_missing(tmp_path):
    cfg=config(tmp_path);raw=Path(cfg.paths.raw);raw.mkdir(parents=True)
    (raw/'a.md').write_text(article('部署'))
    (raw/'b.md').write_text(article('查询'))
    (raw/'c.txt').write_text('太短')
    (raw/'d.xlsx').write_bytes(b'unsupported')
    (raw/'e.md').write_bytes(b'\xff\xfe')
    result=run_clean(cfg=cfg)
    assert result['cleaned']==2 and result['empty']==result['unsupported']==result['failed']==1
    with RegistryDB(cfg.paths.registry_db) as db:
        a=db.get(make_doc_id('a.md'))
        assert a['title']=='部署' and a['version']==1 and a['status']=='cleaned'
        doc=CleanDocument.model_validate_json((Path(cfg.paths.clean)/a['doc_id']/'v1/clean.json').read_text())
        assert doc.blocks[0].level==1 and doc.parser=='text'
        assert (Path(cfg.paths.clean)/a['doc_id']/'v1/full.md').read_text().startswith('# 部署')
    (raw/'f.md').write_text((raw/'a.md').read_text())
    result=run_clean(cfg=cfg)
    assert result['duplicate']==1 and result['failed']==1 and result['skipped']==4
    (raw/'a.md').write_text(article('全新安装'))
    (raw/'e.md').write_text(article('修复编码'))
    (raw/'b.md').unlink()
    result=run_clean(cfg=cfg,limit=1)
    assert result['scanned']==result['cleaned']==1 and result['deleted']==0
    with RegistryDB(cfg.paths.registry_db) as db:
        assert db.get(make_doc_id('b.md'))['status']=='cleaned'
        assert db.get(make_doc_id('a.md'))['version']==2
    result=run_clean(cfg=cfg)
    assert result['deleted']==1 and result['cleaned']==2  # failed修复、duplicate原保留文变化后恢复。
    with RegistryDB(cfg.paths.registry_db) as db:
        assert db.get(make_doc_id('e.md'))['version']==2
        assert db.get(make_doc_id('f.md'))['status']=='cleaned'
    (raw/'a.md').write_text(article('第三版本'))
    run_clean(cfg=cfg)
    root=Path(cfg.paths.clean)/make_doc_id('a.md')
    assert not (root/'v1').exists() and (root/'v2').exists() and (root/'v3').exists()
    assert not list(root.glob('.clean-*'))


def test_limit_zero_local_only_and_input_source_symlink(tmp_path):
    cfg=config(tmp_path);raw=Path(cfg.paths.raw);raw.mkdir(parents=True)
    (raw/'a.md').write_text(article('安全读取'))
    (raw/'b.pdf').write_bytes(b'must not upload')
    (raw/'link.md').symlink_to(raw/'a.md')
    assert run_clean(cfg=cfg,limit=0)['scanned']==0
    result=run_clean(cfg=cfg,local_only=True)
    assert result['scanned']==result['cleaned']==1
    with RegistryDB(cfg.paths.registry_db) as db:
        assert len(db.list_documents())==1
    with pytest.raises(ValueError):run_clean(cfg=cfg,limit=-1)


def test_image_artifact_and_markdown_links(tmp_path):
    import numpy as np
    from PIL import Image
    cfg=config(tmp_path);raw=Path(cfg.paths.raw);raw.mkdir(parents=True)
    image=raw/'flow.png';Image.fromarray(np.random.default_rng(1).integers(0,256,(128,128,3),dtype=np.uint8)).save(image)
    (raw/'a.md').write_text(article('图示部署')+'\n![部署流程](flow.png)\n')
    result=run_clean(cfg=cfg,local_only=True)
    assert result['cleaned']==1
    directory=Path(cfg.paths.clean)/make_doc_id('a.md')/'v1'
    doc=CleanDocument.model_validate_json((directory/'clean.json').read_text())
    images=[b for b in doc.blocks if b.type=='image']
    assert images[0].image_path.startswith(f'clean/{doc.doc_id}/v1/images/')
    copied=directory/'images'/Path(images[0].image_path).name
    assert copied.read_bytes()==image.read_bytes() and images[0].meta['context']=='部署流程'
    assert f'](images/{copied.name})' in (directory/'full.md').read_text()


class Cloud:
    def __init__(self,mode='ok'):self.mode=mode;self.resume_seen=None
    async def parse_batch(self,jobs,**kw):
        self.resume_seen=dict(self.resume_batches)
        self.on_batch('actual-batch-id',jobs)
        result={}
        for j in jobs:
            if self.mode=='quota':result[j.doc_id]=QuotaExceeded(-60018);continue
            if self.mode=='timeout':result[j.doc_id]=TimeoutError('poll timeout');continue
            j.out_dir.mkdir(parents=True,exist_ok=True)
            items=[{'type':'text','text':'演示标题','text_level':1,'page_idx':0},
                   {'type':'text','text':'云端解析的有效正文内容。'*30,'page_idx':0}]
            (j.out_dir/'x_content_list.json').write_text(json.dumps(items,ensure_ascii=False))
            (j.out_dir/'full.md').write_text('not used for mapping')
            result[j.doc_id]=j.out_dir
        return result


def test_cloud_registry_batch_id_resume_timeout_and_pending(tmp_path):
    cfg=config(tmp_path);raw=Path(cfg.paths.raw);raw.mkdir(parents=True)
    (raw/'a.pdf').write_bytes(b'synthetic-pdf')
    cloud=Cloud('timeout')
    assert run_clean(cfg=cfg,mineru=cloud)['failed']==1
    with RegistryDB(cfg.paths.registry_db) as db:
        row=db.get(make_doc_id('a.pdf'));assert row['status']=='parsing' and row['mineru_batch_id']=='actual-batch-id'
    cloud=Cloud('ok')
    assert run_clean(cfg=cfg,mineru=cloud)['cleaned']==1
    assert cloud.resume_seen=={make_doc_id('a.pdf'):'actual-batch-id'}
    with RegistryDB(cfg.paths.registry_db) as db:
        assert db.get(make_doc_id('a.pdf'))['version']==1
    (raw/'b.pdf').write_bytes(b'another-pdf')
    assert run_clean(cfg=cfg,mineru=Cloud('quota'))['pending']==1
    with RegistryDB(cfg.paths.registry_db) as db:
        assert db.get(make_doc_id('b.pdf'))['status']=='pending'


def test_missing_token_does_not_interrupt_local_documents(tmp_path,monkeypatch):
    cfg=config(tmp_path);raw=Path(cfg.paths.raw);raw.mkdir(parents=True)
    (raw/'a.md').write_text(article('本地正文'))
    (raw/'b.pdf').write_bytes(b'pdf')
    monkeypatch.delenv('MINERU_TOKEN',raising=False)
    result=run_clean(cfg=cfg)
    assert result['cleaned']==1 and result['failed']==1


def test_render_formula_table_footnote_and_code():
    out=render_markdown([Block(type='equation',text=r'\frac{a}{b}'),Block(type='table',text='|列|\n|---|',
                         meta={'table_caption':'表一','table_footnote':'注释'}),Block(type='code',text='```\na=1\n```')])
    assert '$$\n\\frac{a}{b}\n$$' in out and '表一\n\n|列|' in out and '注释' in out and '```\na=1\n```' in out


def test_cross_root_collision_does_not_change_original_record(tmp_path):
    cfg=config(tmp_path);first=tmp_path/'first';second=tmp_path/'second'
    first.mkdir();second.mkdir()
    (first/'a.md').write_text(article('原文件'))
    (second/'a.md').write_text(article('另一个目录'))
    assert run_clean(first,cfg=cfg)['cleaned']==1
    with RegistryDB(cfg.paths.registry_db) as db:
        before=db.get(make_doc_id('a.md'))
    assert run_clean(second,cfg=cfg)['failed']==1
    with RegistryDB(cfg.paths.registry_db) as db:
        assert db.get(make_doc_id('a.md'))==before


def test_raw_file_changed_during_cloud_parse_rejects_wrong_version(tmp_path):
    cfg=config(tmp_path);raw=Path(cfg.paths.raw);raw.mkdir(parents=True)
    (raw/'a.pdf').write_bytes(b'source before upload')
    class Changed(Cloud):
        async def parse_batch(self,jobs,**kw):
            result=await super().parse_batch(jobs,**kw)
            for j in jobs:j.path.write_bytes(b'changed after upload')
            return result
    assert run_clean(cfg=cfg,mineru=Changed())['failed']==1
    with RegistryDB(cfg.paths.registry_db) as db:
        assert '变化' in db.get(make_doc_id('a.pdf'))['error']
    assert not (Path(cfg.paths.clean)/make_doc_id('a.pdf')/'v1/clean.json').exists()


def test_cli_suppresses_transfer_request_urls(tmp_path, monkeypatch, caplog, capsys):
    import logging
    import sys
    from ragagent.ingest import pipeline
    cfg = config(tmp_path)
    signed_url = 'https://transfer.example/upload?signature=should-not-be-logged'
    monkeypatch.setattr(sys, 'argv', ['clean', '--local-only'])
    monkeypatch.setattr(pipeline, 'load_config', lambda _: cfg)
    levels = {name: logging.getLogger(name).level for name in ('httpx', 'httpcore')}
    for name in levels:
        monkeypatch.setattr(logging.getLogger(name), 'level', logging.INFO)
    def run(*args, **kwargs):
        assert kwargs['local_only'] is True
        for name in levels:
            logging.getLogger(name).info('HTTP request: %s', signed_url)
        return {'cleaned': 0}
    monkeypatch.setattr(pipeline, 'run_clean', run)
    with caplog.at_level(logging.INFO):
        pipeline.main()
    assert 'signature' not in caplog.text
    assert json.loads(capsys.readouterr().out) == {'cleaned': 0}
