"""仅缓存预处理成功结果；key覆盖问题/历史/Prompt版本/主备模型与场景配置。"""
import hashlib
import json
import os
from pathlib import Path


class PreprocessCache:
    def __init__(self, cfg, call, directory):
        self.cfg,self.call,self.directory=cfg,call,Path(directory)
        self.hits,self.misses=0,0

    async def __call__(self, scene, eps, schema, messages):
        if schema.__name__!='Preprocessed':
            return await self.call(scene,eps,schema,messages)
        metadata={'messages':[m.model_dump(mode='json') for m in messages],
                  'prompt_version':self.cfg.prompt.version,'preprocess':self.cfg.preprocess.model_dump(mode='json'),
                  'scene':scene.model_dump(mode='json'),
                  'endpoints':{name:eps[name].model_dump(mode='json') for name in (scene.primary,scene.fallback)}}
        key=hashlib.sha256(json.dumps(metadata,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        path=self.directory/(key+'.json')
        if path.exists():
            saved=json.loads(path.read_text())
            self.hits+=1
            return schema.model_validate(saved['parsed']),saved['endpoint']
        self.misses+=1
        parsed,endpoint=await self.call(scene,eps,schema,messages)
        if parsed is not None:
            self.directory.mkdir(parents=True,exist_ok=True)
            # tempfile防中断写半个JSON；不缓存故障导致的None。
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w',dir=self.directory,delete=False) as f:
                json.dump({'parsed':parsed.model_dump(mode='json'),'endpoint':endpoint},f,ensure_ascii=False)
                tmp=f.name
            os.replace(tmp,path)
        return parsed,endpoint
