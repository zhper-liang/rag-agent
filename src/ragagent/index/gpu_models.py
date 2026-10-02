"""真实 Visualized-BGE 与 reranker 后端；仅服务进程导入 GPU 依赖。"""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path
import sys

from ragagent.config import Config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class GPUModels:
    def __init__(self, cfg: Config, *, visual_code: Path = Path('models/FlagEmbedding/research/visual_bge'),
                 text_model: Path = Path('models/BAAI/bge-m3'),
                 rerank_model: Path = Path('models/BAAI/bge-reranker-v2-m3')):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = cfg.index.device
        self.dim = cfg.index.dim
        self.batch_size = cfg.index.text_batch
        self.dtype = torch.float16 if cfg.index.fp16 else torch.float32
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA不可用，真实编码服务不能以桩替代')
        weight = Path(cfg.index.embed_weight)
        for path in (weight, text_model / 'config.json', rerank_model / 'model.safetensors',
                     visual_code / 'visual_bge/modeling.py'):
            if not path.is_file():
                raise FileNotFoundError(f'真实模型或官方研究代码缺失：{path}')
        sys.path.insert(0, str(visual_code.resolve()))
        from visual_bge.modeling import Visualized_BGE

        dtype = self.dtype

        class InferenceModel(Visualized_BGE):
            def load_model(model, model_weight):
                # 官方构造器随后会移到CUDA。先在CPU加载并转换，避免fp32瞬时显存峰值。
                state = torch.load(model_weight, map_location='cpu', weights_only=True, mmap=True)
                model.load_state_dict(state, strict=True, assign=True)
                model.to(dtype=dtype)

        self.model_name = 'visualized-bge-m3'
        weight_hash = _sha256(weight)
        self.version = f'{self.model_name}@sha256:{weight_hash}'
        self.visual = InferenceModel(model_name_bge=cfg.index.embed_model, model_weight=str(weight),
                                     from_pretrained=str(text_model))
        self.visual.to(device=self.device, dtype=self.dtype).eval()
        self.visual.device = torch.device(self.device)
        self.visual.dtype = self.dtype
        self.visual.requires_grad_(False)
        self.tokenizer = self.visual.tokenizer
        self.tokenizer.truncation_side = 'right'  # 保留embed_text开头的标题路径
        positions = self.visual.bge_embeddings.position_embeddings.num_embeddings
        padding = self.visual.bge_embeddings.padding_idx
        self.text_max_length = positions - padding - 1
        patch_embed = self.visual.model_visual.visual.patch_embed
        self.image_tokens = patch_embed.num_patches
        self.image_text_max_length = self.text_max_length - self.image_tokens
        if self.dim != self.visual.hidden_dim:
            raise ValueError('Visualized-BGE维度与index.dim不一致')
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model, local_files_only=True)
        self.reranker = AutoModelForSequenceClassification.from_pretrained(
            rerank_model, local_files_only=True, dtype=self.dtype, attn_implementation='sdpa')
        self.reranker.to(self.device).eval().requires_grad_(False)
        self.max_length = self.reranker.config.max_position_embeddings - self.reranker.config.pad_token_id - 1
        self.rerank_version = f'{cfg.rerank.model}@sha256:{_sha256(rerank_model / "model.safetensors")}'
        if self.device.startswith('cuda'):
            torch.cuda.synchronize(self.device)

    def _rows(self, tensor):
        tensor = self.torch.nn.functional.normalize(tensor.float(), dim=-1)
        if not self.torch.isfinite(tensor).all():
            raise RuntimeError('编码模型产生非有限向量')
        return tensor.cpu().tolist()

    def _text_batch(self, texts):
        torch = self.torch
        try:
            with torch.inference_mode():
                inputs = self.tokenizer(texts, return_tensors='pt', padding=True,
                                        truncation=True, max_length=self.text_max_length).to(self.device)
                return self._rows(self.visual.encode_text(inputs))
        except torch.cuda.OutOfMemoryError:
            # OOM后仅缩小当前批次；不改共享配置，保持输入/输出顺序。
            if len(texts) == 1:
                raise
        # 离开except后释放失败推理的traceback，再重试，避免保留激活。
        gc.collect()
        torch.cuda.empty_cache()
        middle = len(texts) // 2
        return self._text_batch(texts[:middle]) + self._text_batch(texts[middle:])

    def text(self, texts: list[str]) -> list[list[float]]:
        # 按长度聚合减少padding，返回时恢复调用者的次序。
        ordered = sorted(enumerate(texts), key=lambda item: len(item[1]))
        rows = [None] * len(texts)
        for offset in range(0, len(ordered), self.batch_size):
            batch = ordered[offset:offset + self.batch_size]
            vectors = self._text_batch([text for _, text in batch])
            for (index, _), vector in zip(batch, vectors, strict=True):
                rows[index] = vector
        return rows

    def image(self, items: list[tuple[str, str]]) -> list[list[float]]:
        from PIL import Image
        torch = self.torch
        rows = []
        # 图文官方示例逐条编码；图像patch占用的位置从文本上限中扣除。
        with torch.inference_mode():
            for path, text in items:
                with Image.open(path) as source:
                    image = self.visual.preprocess_val(source.convert('RGB')).unsqueeze(0).to(
                        device=self.device, dtype=self.dtype)
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                        max_length=self.image_text_max_length).to(self.device)
                rows.extend(self._rows(self.visual.encode_mm(image, inputs)))
        return rows

    def rerank(self, pairs: list[tuple[str, str]], max_length: int) -> list[float]:
        if not 4 <= max_length <= self.max_length:
            raise ValueError('重排max_length超出模型范围')
        scores = []
        with self.torch.inference_mode():
            for query, passage in pairs:
                inputs = self.rerank_tokenizer(query, passage, return_tensors='pt',
                                              truncation=True, max_length=max_length).to(self.device)
                logits = self.reranker(**inputs).logits.reshape(-1).float()
                if logits.numel() != 1 or not self.torch.isfinite(logits).all():
                    raise RuntimeError('重排模型输出无效')
                scores.append(self.torch.sigmoid(logits).item())
        return scores

    def health(self) -> dict:
        report = {'model': self.model_name, 'version': self.version, 'device': self.device,
                  'dim': self.dim, 'dtype': str(self.dtype), 'text_max_length': self.text_max_length,
                  'image_tokens': self.image_tokens, 'image_text_max_length': self.image_text_max_length,
                  'rerank_model': self.rerank_version, 'rerank_max_length': self.max_length}
        if self.device.startswith('cuda'):
            torch = self.torch
            report['gpu_memory_mib'] = {
                'allocated': round(torch.cuda.memory_allocated(self.device) / 2 ** 20, 2),
                'reserved': round(torch.cuda.memory_reserved(self.device) / 2 ** 20, 2),
                'peak_allocated': round(torch.cuda.max_memory_allocated(self.device) / 2 ** 20, 2),
                'peak_reserved': round(torch.cuda.max_memory_reserved(self.device) / 2 ** 20, 2)}
        return report

    def close(self):
        self.visual = None
        self.reranker = None
        gc.collect()
        if self.device.startswith('cuda'):
            self.torch.cuda.empty_cache()
