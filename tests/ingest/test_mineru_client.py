"""真实协议形状的HTTP MockTransport，明确不作为云端验收。"""
import asyncio
from io import BytesIO
import json
import stat
import zipfile

import httpx
import pytest
from pydantic import ValidationError
from ragagent.config import load_config
from ragagent.ingest.mineru.client import MinerUClient, ParseJob, MinerUError, QuotaExceeded, extract_zip
from ragagent.ingest.dedup import sha256


def archive(extra=None):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer,'w') as z:
        z.writestr('sample/full.md','# Sample\n\n正文')
        z.writestr('sample/x_content_list.json',json.dumps([{'type':'text','text':'有效正文'}]))
        for name,content in extra or []:
            z.writestr(name,content)
    return buffer.getvalue()


def job(tmp_path,number=1):
    path=tmp_path/f'{number}.pdf';path.write_bytes(b'local source')
    return ParseJob(doc_id=f'{number:016x}',path=path,out_dir=tmp_path/'parsed'/f'{number:016x}'/'v1')


def cfg(**updates):
    return load_config().ingest.mineru.model_copy(update={'base_url':'https://api.test/api/v4','retries':0,**updates})


def test_submit_upload_poll_fetch_and_validated_cache(tmp_path):
    jobs=[job(tmp_path,1),job(tmp_path,2)]
    calls, callbacks, polls=[],[],0
    def handler(req):
        nonlocal polls
        calls.append((req.method,req.url.host,req.url.path))
        if req.url.host=='api.test':
            assert req.headers['Authorization']=='Bearer secret'
            if req.method=='POST':
                payload=json.loads(req.content)
                assert payload['model_version']=='vlm' and payload['enable_table'] and payload['enable_formula']
                assert payload['files']==[{'name':j.path.name,'data_id':j.doc_id,'is_ocr':False} for j in jobs]
                assert 'is_ocr' not in {k for k in payload if k!='files'}
                return httpx.Response(200,json={'code':0,'data':{'batch_id':'b1','file_urls':['https://upload.test/1','https://upload.test/2']}})
            polls+=1
            state='running' if polls==1 else 'done'
            return httpx.Response(200,json={'code':0,'data':{'extract_result':[{'data_id':j.doc_id,'state':state,'full_zip_url':f'https://download.test/{j.doc_id}'} for j in jobs]}})
        assert 'authorization' not in req.headers and 'content-type' not in req.headers
        if req.method=='PUT':
            assert req.content==b'local source'
            return httpx.Response(200)
        return httpx.Response(200,content=archive())
    # 最后一次重试只提交一个job，handler前半截断言也需针对请求数量。
    def flexible(req):
        if req.method=='POST' and len(json.loads(req.content)['files'])==1:
            assert 'Authorization' in req.headers
            return httpx.Response(200,json={'code':0,'data':{'batch_id':'b2','file_urls':['https://upload.test/1']}})
        return handler(req)
    # 保留对第一轮实际payload的严格检查，后续再请求独立恢复。
    async def run():
        async with MinerUClient('secret',cfg=cfg(),transport=httpx.MockTransport(flexible),
                                on_batch=lambda batch,jobs:callbacks.append((batch,len(jobs)))) as client:
            results=await client.parse_batch(jobs,poll_interval=0,timeout=1)
            assert all(isinstance(r,type(tmp_path)) for r in results.values())
            count=len(calls)
            assert await client.parse_batch(jobs,poll_interval=0,timeout=1)==results and len(calls)==count
            (jobs[0].out_dir/'sample/full.md').write_text('tampered')
            renewed=await client.parse_batch([jobs[0]],poll_interval=0,timeout=1)
            assert isinstance(renewed[jobs[0].doc_id],type(tmp_path))
    asyncio.run(run())
    assert callbacks==[('b1',2),('b2',1)] and polls>=2
    assert (jobs[0].out_dir/'sample/full.md').read_text().startswith('# Sample')


def test_resume_batch_does_not_submit_again(tmp_path):
    j=job(tmp_path)
    def handler(req):
        assert req.method=='GET'
        if req.url.host=='api.test':
            assert req.url.path.endswith('/resume-batch')
            return httpx.Response(200,json={'code':0,'data':{'extract_result':[{'data_id':j.doc_id,'state':'done','full_zip_url':'https://download.test/file'}]}})
        assert 'authorization' not in req.headers
        return httpx.Response(200,content=archive())
    async def run():
        async with MinerUClient('secret',cfg=cfg(),transport=httpx.MockTransport(handler),resume_batches={j.doc_id:'resume-batch'}) as client:
            result=await client.parse_batch([j],poll_interval=0,timeout=1)
            assert result[j.doc_id]==j.out_dir
    asyncio.run(run())


def test_individual_failed_and_timeout_are_not_success(tmp_path):
    jobs=[job(tmp_path,n) for n in (1,2)]
    def handler(req):
        if req.method=='POST':
            return httpx.Response(200,json={'code':0,'data':{'batch_id':'batch','file_urls':['https://upload.test/1','https://upload.test/2']}})
        if req.method=='PUT':
            return httpx.Response(200)
        return httpx.Response(200,json={'code':0,'data':{'extract_result':[{'data_id':jobs[0].doc_id,'state':'failed','err_code':-60007},{'data_id':jobs[1].doc_id,'state':'pending'}]}})
    async def run():
        async with MinerUClient('secret',cfg=cfg(),transport=httpx.MockTransport(handler)) as client:
            result=await client.parse_batch(jobs,poll_interval=.01,timeout=.03)
            assert isinstance(result[jobs[0].doc_id],MinerUError)
            assert isinstance(result[jobs[1].doc_id],TimeoutError)
    asyncio.run(run())


def test_quota_stops_new_submissions_and_http_200_code_checked(tmp_path):
    jobs=[job(tmp_path,n) for n in (1,2,3)]
    submits=[]
    def handler(req):
        submits.append(req)
        return httpx.Response(200,json={'code':-60018,'msg':'quota'})
    async def run():
        async with MinerUClient('secret',cfg=cfg(batch_size=1,concurrency=1),transport=httpx.MockTransport(handler)) as client:
            result=await client.parse_batch(jobs,poll_interval=0,timeout=1)
            assert all(isinstance(r,QuotaExceeded) for r in result.values())
    asyncio.run(run())
    assert len(submits)==1


def test_network_5xx_retry_and_upload_response_length(tmp_path,monkeypatch):
    j=job(tmp_path);calls=0;sleeps=[]
    async def sleep(seconds):sleeps.append(seconds)
    monkeypatch.setattr('ragagent.ingest.mineru.client._sleep',sleep)
    def handler(req):
        nonlocal calls
        calls+=1
        if calls==1:raise httpx.ConnectError('signed url secret',request=req)
        if calls==2:return httpx.Response(503)
        return httpx.Response(200,json={'code':0,'data':{'batch_id':'batch','file_urls':[]}})
    async def run():
        async with MinerUClient('secret',cfg=cfg(retries=2),transport=httpx.MockTransport(handler)) as client:
            result=await client.parse_batch([j],poll_interval=0,timeout=1)
            assert isinstance(result[j.doc_id],MinerUError)
            assert 'invalid-upload-response' in str(result[j.doc_id])
    original_sleep = asyncio.sleep
    asyncio.run(run())
    assert asyncio.sleep is original_sleep
    assert calls==3 and sleeps==[1,2]


@pytest.mark.parametrize('name',['../escape','/absolute','C:/windows','a\\..\\x'])
def test_zip_traversal_rejected(tmp_path,name):
    p=tmp_path/'result.zip';p.write_bytes(archive([(name,b'bad')]))
    with pytest.raises(ValueError):extract_zip(p,tmp_path/'out',source_hash='x')
    assert not (tmp_path/'out').exists() and not list(tmp_path.glob('.mineru-*'))


def test_zip_symlink_incomplete_duplicate_rejected(tmp_path):
    p=tmp_path/'result.zip'
    with zipfile.ZipFile(p,'w') as z:
        link=zipfile.ZipInfo('link');link.create_system=3;link.external_attr=(stat.S_IFLNK|0o777)<<16
        z.writestr(link,'/etc/passwd')
    with pytest.raises(ValueError):extract_zip(p,tmp_path/'out',source_hash='x')
    with zipfile.ZipFile(p,'w') as z:z.writestr('full.md','no content list')
    with pytest.raises(ValueError):extract_zip(p,tmp_path/'out',source_hash='x')
    with pytest.warns(UserWarning, match='Duplicate name'):
        p.write_bytes(archive([('sample/full.md',b'duplicate')]))
    with pytest.raises(ValueError):extract_zip(p,tmp_path/'out',source_hash='x')


def test_job_and_duplicate_id_validation(tmp_path):
    with pytest.raises(ValidationError):ParseJob(doc_id='../bad',path=tmp_path,out_dir=tmp_path)
    j=job(tmp_path)
    async def run():
        async with MinerUClient('secret',cfg=cfg()) as client:
            with pytest.raises(ValueError):await client.parse_batch([j,j])
    asyncio.run(run())


def test_source_changed_while_remote_parsing_not_cached_as_new_content(tmp_path):
    j=job(tmp_path)
    def handler(req):
        if req.method=='POST':
            return httpx.Response(200,json={'code':0,'data':{'batch_id':'batch','file_urls':['https://upload.test/1']}})
        if req.method=='PUT':
            j.path.write_bytes(b'changed after upload');return httpx.Response(200)
        return httpx.Response(200,json={'code':0,'data':{'extract_result':[{'data_id':j.doc_id,'state':'done','full_zip_url':'https://download.test/file'}]}})
    async def run():
        async with MinerUClient('secret',cfg=cfg(),transport=httpx.MockTransport(handler)) as client:
            result=await client.parse_batch([j],poll_interval=0,timeout=1)
            assert isinstance(result[j.doc_id],ValueError) and '变化' in str(result[j.doc_id])
            assert not j.out_dir.exists()
    asyncio.run(run())
