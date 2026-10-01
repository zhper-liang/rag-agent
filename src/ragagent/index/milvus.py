"""Milvus连接、索引就绪等待和别名切换；不管理容器生命周期。"""
from __future__ import annotations

import os
import time

from pymilvus import MilvusClient

from ragagent.config import Config, load_config
from ragagent.index.schema import create_collection

INDEXED = ('dense', 'sparse', 'doc_id', 'type', 'file_type', 'searchable')


def connect(cfg: Config | None = None) -> MilvusClient:
    service = (cfg or load_config()).services.milvus
    return MilvusClient(uri=service.uri, user=os.environ.get(service.user_env, ''),
                        password=os.environ.get(service.password_env, ''))


def ensure_collection(client, name: str, dim: int) -> bool:
    """只在新建集合上配置mmap，已有集合不release、不改属性。"""
    if client.has_collection(name):
        schema = client.describe_collection(name)
        dense = next(f for f in schema['fields'] if f['name'] == 'dense')
        if int(dense['params']['dim']) != dim:
            raise ValueError('现有collection维度与index.dim不一致')
        return False
    create_collection(client, name, dim)
    # 共享schema创建时自动load；改索引属性前只release刚创建的collection。
    client.release_collection(name)
    client.alter_collection_properties(name, properties={'mmap.enabled': True})
    for field in INDEXED[1:]:
        client.alter_index_properties(name, field, properties={'mmap.enabled': True})
    # Milvus过滤删除要求collection已load；先load空集合供文档增量写入。
    # 验收仍须flush后indexed_rows==total_rows>0，并再次load之后才查询。
    load(client, name)
    return True


def wait_indexed(client, coll: str, *, timeout_s: float = 600,
                 poll_s: float = 2) -> dict:
    deadline = time.monotonic() + timeout_s
    while True:
        indexes = {f: client.describe_index(coll, f) for f in INDEXED}
        if all(d.get('indexed_rows') == d.get('total_rows', 0) > 0
               for d in indexes.values()):
            return indexes
        if any(str(d.get('state', '')).lower() == 'failed' for d in indexes.values()):
            raise RuntimeError(f'索引构建失败：{indexes}')
        if time.monotonic() >= deadline:
            progress = {f: (d.get('indexed_rows'), d.get('total_rows')) for f, d in indexes.items()}
            raise TimeoutError(f'索引未建完，不能以growing暴力查询冒充DISKANN：{progress}')
        time.sleep(poll_s)


def load(client, coll: str, *, timeout_s: float = 120):
    client.load_collection(coll, timeout=timeout_s)
    deadline = time.monotonic() + timeout_s
    while 'Loaded' not in str(client.get_load_state(coll)['state']):
        if time.monotonic() >= deadline:
            raise TimeoutError('collection加载超时')
        time.sleep(0.5)


def switch_alias(client, coll: str, alias: str = 'rag_chunks'):
    """只有调用方验收通过后才调用；M1a绝不切换正式别名。"""
    try:
        client.describe_alias(alias)
    except Exception:
        # 不吞掉网络/认证错误：list_aliases再确认别名确实不存在。
        if alias in client.list_aliases():
            raise
        client.create_alias(collection_name=coll, alias=alias)
    else:
        client.alter_alias(collection_name=coll, alias=alias)
