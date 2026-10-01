import json
from pathlib import Path
import subprocess
import pymupdf
import pytest
from ragagent.ingest.mineru.split import split_pdf, merge_parts, office_to_pdf, PDFPart
from ragagent.ingest.dedup import sha256
from tests.ingest.test_pipeline import config, run_clean, Cloud, RegistryDB, make_doc_id


def pdf(path,pages=5):
    with pymupdf.open() as document:
        for i in range(pages):
            page=document.new_page();page.insert_text((72,72),f'Page {i+1}: heading, formula a+b, table fields.')
        document.save(path)


def test_pdf_page_byte_limits_and_readonly_source(tmp_path):
    source=tmp_path/'a.pdf';pdf(source)
    before=sha256(source)
    parts=split_pdf(source,tmp_path/'split',max_pages=2)
    assert [(p.start_page,p.page_count) for p in parts]==[(0,2),(2,2),(4,1)]
    assert sha256(source)==before
    assert [p.path.read_bytes() for p in split_pdf(source,tmp_path/'split',max_pages=2)]==[p.path.read_bytes() for p in parts]
    assert all(p.path.stat().st_size<200*1024*1024 for p in parts)
    unchanged=split_pdf(source,tmp_path/'not-created',max_pages=200)
    assert unchanged[0].path==source and not (tmp_path/'not-created').exists()
    with pytest.raises(ValueError,match='单页'):split_pdf(source,tmp_path/'tiny',max_bytes=1)
    with pytest.raises(ValueError):split_pdf(source,tmp_path/'split',max_pages=0)


def test_merge_page_offsets_images_and_gap_rejected(tmp_path):
    from PIL import Image
    source=tmp_path/'a.pdf';pdf(source)
    parts=split_pdf(source,tmp_path/'input',max_pages=2)
    parsed=[]
    for i,part in enumerate(parts):
        root=tmp_path/f'parsed-{i}';(root/'images').mkdir(parents=True)
        Image.new('RGB',(100,100),(20*i,100,200)).save(root/'images'/'image.png')
        (root/'x_content_list.json').write_text(json.dumps([
            {'type':'text','text':f'part {i}','page_idx':0},
            {'type':'image','img_path':'images/image.png','image_caption':['diagram'],'page_idx':0}]))
        parsed.append((part,root))
    out=merge_parts(parsed,tmp_path/'merged',source_path=source)
    items=json.loads((out/'merged_content_list.json').read_text())
    assert [i['page_idx'] for i in items]==[0,0,2,2,4,4]
    assert len(list((out/'images').glob('*')))==3
    assert all((out/i['img_path']).is_file() for i in items if 'img_path' in i)
    assert (out/'_complete.json').is_file() and '](' in (out/'full.md').read_text()
    with pytest.raises(ValueError,match='缺页'):merge_parts(parsed[1:],tmp_path/'missing',source_path=source)


def test_office_conversion_args_no_shell_and_private_profile(tmp_path,monkeypatch):
    source=tmp_path/'unsafe name.docx';source.write_bytes(b'office')
    out=tmp_path/'converted';calls=[]
    monkeypatch.setattr('ragagent.ingest.mineru.split.shutil.which',lambda _: '/usr/bin/libreoffice')
    def run(args,**kwargs):
        calls.append((args,kwargs));pdf(out/'unsafe name.pdf',1)
        return subprocess.CompletedProcess(args,0,b'',b'')
    monkeypatch.setattr('ragagent.ingest.mineru.split.subprocess.run',run)
    assert office_to_pdf(source,out)==out/'unsafe name.pdf'
    args,kwargs=calls[0]
    assert args[-1]==str(source) and '--headless' in args and any(a.startswith('-env:UserInstallation=file:') for a in args)
    assert 'shell' not in kwargs


def test_201_page_pipeline_split_resumes_each_part(tmp_path):
    cfg=config(tmp_path);raw=Path(cfg.paths.raw);raw.mkdir(parents=True)
    pdf(raw/'large.pdf',201)
    cloud=Cloud('timeout')
    assert run_clean(cfg=cfg,mineru=cloud)['failed']==1
    assert len(cloud.resume_seen)==0
    with RegistryDB(cfg.paths.registry_db) as db:
        rows=db.conn.execute('SELECT * FROM ingest_parse_parts').fetchall()
        assert len(rows)==2 and all(r['batch_id']=='actual-batch-id' for r in rows)
        assert db.get(make_doc_id('large.pdf'))['status']=='parsing'
    cloud=Cloud('ok')
    assert run_clean(cfg=cfg,mineru=cloud)['cleaned']==1
    assert len(cloud.resume_seen)==2
    out=Path(cfg.paths.clean)/make_doc_id('large.pdf')/'v1/clean.json'
    doc=json.loads(out.read_text())
    assert doc['meta']['pages']==201
    # 同样两页标题/正文可能被页重复清洗，pages仍来自输入映射完整201页。
    with RegistryDB(cfg.paths.registry_db) as db:
        assert db.get(make_doc_id('large.pdf'))['version']==1


def test_pdf_registry_pages_includes_blank_trailing_page(tmp_path):
    cfg=config(tmp_path);raw=Path(cfg.paths.raw);raw.mkdir(parents=True)
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72,72),'Content');doc.new_page();doc.save(raw/'two.pdf')
    assert run_clean(cfg=cfg,mineru=Cloud())['cleaned']==1
    with RegistryDB(cfg.paths.registry_db) as db:
        assert db.get(make_doc_id('two.pdf'))['pages']==2
