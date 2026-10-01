import sqlite3
import pytest
from ragagent.ingest.registry_db import RegistryDB, connect_db, make_doc_id


def test_registry_schema_fts_incremental_and_sync(tmp_path):
    with RegistryDB(tmp_path/'registry.db') as db:
        columns = [r[1] for r in db.conn.execute('PRAGMA table_info(documents)')]
        assert columns == ['doc_id','source_path','title','file_type','parser','content_hash','version','status',
                           'mineru_batch_id','pages','duplicate_of','error','updated_at']
        a, changed = db.register('目录/部署手册.md','a'*64,title='部署指导手册')
        assert changed and a['version']==1
        same,changed = db.register('目录/部署手册.md','a'*64)
        assert not changed and same==a
        db.set_status(a['doc_id'],'indexed',pages=3)
        assert db.title_candidates('《部署指导手册.md》')[0]['doc_id']==a['doc_id']
        assert db.title_candidates('部署手')[0]['filename']=='部署手册'
        changed,yes = db.register('目录/部署手册.md','b'*64)
        assert yes and changed['version']==2 and changed['status']=='pending'
        assert not db.title_candidates('部署手册')
        db.update(a['doc_id'],title='改名文档',status='indexed')
        assert db.title_candidates('改名文档') and not db.title_candidates('部署指导手册')
        assert db.conn.execute('SELECT count(*) FROM doc_titles').fetchone()[0]==1
        db.delete(a['doc_id'])
        assert db.get(a['doc_id'])['status']=='deleted' and not db.title_candidates('改名文档')
        db.delete(a['doc_id'],purge=True)
        assert db.get(a['doc_id']) is None
        assert db.conn.execute('SELECT count(*) FROM doc_titles').fetchone()[0]==0


def test_short_query_exact_and_filename_and_only_indexed():
    with RegistryDB(':memory:') as db:
        a,_ = db.register('手册.md','1',title='其他标题')
        b,_ = db.register('长标题.md','2',title='手册其他')
        db.set_status(a['doc_id'],'indexed')
        db.set_status(b['doc_id'],'indexed')
        assert len(db.title_candidates('手册'))==1
        assert not db.title_candidates('标题')
        assert not db.title_candidates('')
        assert not db.title_candidates('" OR * --')


def test_resume_and_deleted_reappearance_and_duplicates():
    with RegistryDB(':memory:') as db:
        a,_ = db.register('a.md','same')
        db.set_status(a['doc_id'],'parsing',mineru_batch_id='batch')
        assert db.resumable()[0]['mineru_batch_id']=='batch'
        db.set_status(a['doc_id'],'cleaned')
        b,_ = db.register('b.md','same')
        assert db.exact_duplicate('same',exclude_doc_id=b['doc_id'])['doc_id']==a['doc_id']
        assert db.mark_missing({'b.md'})==[a['doc_id']]
        restored,changed = db.register('a.md','same')
        assert changed and restored['version']==1 and restored['status']=='pending'


def test_safe_update_and_transaction_rollback():
    with RegistryDB(':memory:') as db:
        a,_ = db.register('a.md','h')
        with pytest.raises(ValueError):
            db.update(a['doc_id'],**{'status = "indexed" --':'x'})
        with pytest.raises(ValueError):
            db.set_status(a['doc_id'],'invalid')
        with pytest.raises(ValueError):
            db.get('bad-id')
        with pytest.raises(sqlite3.IntegrityError):
            db.update(a['doc_id'],version=0,title='should rollback')
        assert db.get(a['doc_id'])['title']=='a'
        assert db.conn.execute('SELECT title FROM doc_titles').fetchone()[0]=='a'
        with pytest.raises(ValueError):
            db.title_candidates('a',limit=-1)


def test_readonly_connection_and_idempotent_init(tmp_path):
    path = tmp_path/'r.db'
    with RegistryDB(path) as db:
        a,_ = db.register('sub/a.md','h')
        db.set_status(a['doc_id'],'indexed')
    conn = connect_db(path)
    conn.close()
    with RegistryDB(path,readonly=True) as db:
        assert db.get_connection().execute('SELECT count(*) FROM documents').fetchone()[0]==1
        with pytest.raises(sqlite3.OperationalError):
            db.register('b.md','h2')
    assert make_doc_id('sub\\a.md')==make_doc_id('sub/a.md')
