"""bge-m3 原版 tokenizer 的无 special-token 计数，带有界 LRU 缓存。

从 models/{model}/tokenizer.json 加载；缺文件时下载，仅初始化时做网络 I/O。
不会导入 torch 或加载编码权重。模型名称读取 config.yaml.chunking.tokenizer。
"""
from functools import lru_cache
from pathlib import Path
import argparse
import json

import yaml
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer as FastTokenizer

from ragagent.config import Chunking, DEFAULT_PATH


class TokenCounter:
    def __init__(self,model: str, *, path: str | Path | None=None, revision=None, endpoint=None):
        self.model = model
        path = Path(path) if path else Path('models') / model / 'tokenizer.json'
        if not path.is_file():
            downloaded = hf_hub_download(repo_id=model,filename='tokenizer.json',
                revision=revision,local_dir=str(path.parent),endpoint=endpoint)
            path = Path(downloaded)
        self.path = path
        self.tokenizer = FastTokenizer.from_file(str(path))
        # tokenizer.json 中若有默认 truncation/padding，预算计数必须取消。
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    @lru_cache(maxsize=8192)
    def count(self,text: str) -> int:
        if not isinstance(text,str):
            raise TypeError('tokenizer 输入必须为 str')
        return len(self.tokenizer.encode(text,add_special_tokens=False).ids)

    __call__ = count

    def cache_info(self):
        return self.count.cache_info()

    def clear_cache(self):
        self.count.cache_clear()


@lru_cache(maxsize=4)
def get_counter(model: str | None=None) -> TokenCounter:
    if model is None:
        cfg = Chunking.model_validate(yaml.safe_load(DEFAULT_PATH.read_text(encoding='utf-8'))['chunking'])
        model = cfg.tokenizer
    return TokenCounter(model)


def count_tokens(text: str) -> int:
    """L3 可直接用作 count 参数；不含 BOS/EOS，不截断，Unicode 原样计数。"""
    return get_counter().count(text)


count = count_tokens


def main():
    parser=argparse.ArgumentParser(description='bge-m3 tokenizer计数/下载校验')
    parser.add_argument('text',nargs='?',default='部署 Milvus DISKANN 索引')
    args=parser.parse_args()
    counter=get_counter()
    print(json.dumps({'model':counter.model,'path':str(counter.path),'tokens':counter(args.text),
                      'special_tokens':False,'cache':counter.cache_info()._asdict()},ensure_ascii=False))


if __name__=='__main__':
    main()
