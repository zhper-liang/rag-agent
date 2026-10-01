"""标准只读评测依赖（不修改线上collection/alias/登记表，不启用Langfuse）。"""
from contextlib import asynccontextmanager
import os

from pymilvus import MilvusClient

from ragagent.graph.deps import Deps
from ragagent.ingest.chunking.tokenizer import get_counter
from ragagent.ingest.registry_db import RegistryDB


@asynccontextmanager
async def open_deps(cfg, collection):
    cfg=cfg.model_copy(update={'services':cfg.services.model_copy(update={
        'milvus':cfg.services.milvus.model_copy(update={'collection':collection})})})
    svc=cfg.services.milvus
    registry=RegistryDB(cfg.paths.registry_db,readonly=True)
    milvus=None
    try:
        milvus=MilvusClient(uri=svc.uri,user=os.environ.get(svc.user_env,''),password=os.environ.get(svc.password_env,''))
        yield Deps(cfg=cfg,count=get_counter(cfg.chunking.tokenizer),registry=registry,milvus=milvus)
    finally:
        registry.close()
        if milvus is not None:
            milvus.close()
        from ragagent.llm.client import close_clients
        await close_clients()
