"""config.yaml → pydantic 模型，含 大模型生成.md §3 的启动校验。

未知字段、重复 key、校验失败都抛 ConfigError；`python -m ragagent.config` 校验后以退出码表示结果。
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError, model_validator

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ConfigError(Exception):
    pass


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


# ───────── paths / services ─────────

class Paths(_Model):
    raw: str
    parsed: str
    clean: str
    chunks: str
    registry_db: str
    session_db: str
    feedback_db: str
    keep_versions: PositiveInt


class Milvus(_Model):
    uri: str
    user_env: str
    password_env: str
    collection: str


class EmbedService(_Model):
    host: str
    port: PositiveInt


class Services(_Model):
    milvus: Milvus
    embed: EmbedService


# ───────── 离线 ─────────

class ParseRoutes(_Model):
    mineru: list[str]
    text: list[str]


class Mineru(_Model):
    base_url: str
    token_env: str
    model_version: Literal["vlm", "pipeline"]
    is_ocr: bool
    enable_formula: bool
    enable_table: bool
    language: str
    batch_size: PositiveInt
    concurrency: PositiveInt
    poll_interval_s: float
    timeout_s: float
    retries: int


class Clean(_Model):
    repeat_page_ratio: float
    heading_demote_chars: int
    table_max_cols: int
    watermarks: list[str]


class Images(_Model):
    min_side_px: int
    min_area_px: int
    repeat_page_ratio: float
    context_chars: int
    llm_description: bool


class Quality(_Model):
    min_block_chars: int
    garbled_ratio: float
    min_doc_chars: int


class IngestDedup(_Model):
    minhash_threshold: float


class Ingest(_Model):
    parse_routes: ParseRoutes
    mineru: Mineru
    clean: Clean
    images: Images
    quality: Quality
    dedup: IngestDedup


class Semantic(_Model):
    enabled: bool
    break_z: float
    strong_z: float
    min_units: int
    min_std: float
    early_flush_ratio: float


class ChunkingDedup(_Model):
    boilerplate_doc_count: int


class Chunking(_Model):
    tokenizer: str
    target_tokens: PositiveInt
    max_tokens: PositiveInt
    min_tokens: PositiveInt
    overlap_tokens: int
    inline_code_tokens: int
    lead_in_tokens: int
    path_max_depth: PositiveInt
    path_heading_chars: PositiveInt
    semantic: Semantic
    dedup: ChunkingDedup


class Index(_Model):
    embed_model: str
    embed_weight: str
    dim: PositiveInt
    device: str
    fp16: bool
    text_batch: PositiveInt
    image_batch: PositiveInt
    insert_batch: PositiveInt


# ───────── 在线：预处理 / 检索 / 上下文 / Prompt ─────────

class History(_Model):
    turns: int
    answer_tokens: int
    total_tokens: int


class FileTypeSynonym(_Model):
    types: list[str]
    words: list[str]


class Scope(_Model):
    fuzzy_threshold: float
    fts_candidates: PositiveInt
    max_docs: PositiveInt
    file_type_synonyms: list[FileTypeSynonym]


class Preprocess(_Model):
    max_question_chars: PositiveInt
    history: History
    scope: Scope


class Rewrite(_Model):
    max_queries: PositiveInt
    hint_candidates: PositiveInt


class Weights(_Model):
    dense: float
    sparse: float


class Retrieval(_Model):
    dense_top_k: PositiveInt
    sparse_top_k: PositiveInt
    search_list: PositiveInt
    weights: Weights
    per_query_limit: PositiveInt
    candidates: PositiveInt
    timeout_ms: PositiveInt

    @model_validator(mode="after")
    def _search_list(self):
        if self.search_list < max(self.dense_top_k, self.sparse_top_k):
            raise ValueError("retrieval.search_list 必须 ≥ dense_top_k / sparse_top_k")
        return self


class Rerank(_Model):
    model: str
    max_length: PositiveInt
    threshold: float
    per_doc_cap: PositiveInt
    top_n: PositiveInt
    timeout_ms: PositiveInt


class Expansion(_Model):
    section_max_tokens: PositiveInt
    images_per_chunk: int
    images_total: int


class Grade(_Model):
    pass_score: float
    max_retries: int


class ContextImage(_Model):
    max_images: int
    tokens_per_image: PositiveInt
    max_side_px: PositiveInt
    jpeg_quality: int = Field(ge=1, le=100)


class Context(_Model):
    budget_tokens: PositiveInt
    block_overhead_tokens: int
    token_safety_factor: float
    image: ContextImage


class Prompt(_Model):
    dir: str
    version: str
    supplement_heading: str
    history_turns: int
    history_answer_tokens: int
    history_total_tokens: int
    max_input_tokens: PositiveInt
    fallback_sources: int


# ───────── 在线：大模型生成（大模型生成.md §3）─────────

class Endpoint(_Model):
    base_url: str
    model: str
    api_key_env: str
    vision: bool
    structured_output: Literal["function_calling", "json_schema", "json_mode"]
    strip_think: bool
    max_concurrency: PositiveInt
    extra_body: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self):
        if not self.base_url.strip() or not self.model.strip():
            raise ValueError("base_url 和 model 不能为空")
        u = urlparse(self.base_url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ValueError(f"base_url 不是合法的 http(s) 地址：{self.base_url!r}")
        if u.scheme != "https" and u.hostname not in LOCAL_HOSTS:
            raise ValueError(f"非本机地址必须是 https：{self.base_url!r}")
        return self


class StreamScene(_Model):
    """generate / chat：首字前重试 + 切换（大模型生成.md §6.1）。"""
    primary: str
    fallback: str
    temperature: float
    max_tokens: PositiveInt
    first_token_timeout_s: float
    pre_first_token_budget_s: float
    total_timeout_s: float
    retries: int
    backoff_s: list[float]


class OneShotScene(_Model):
    """preprocess / rewrite / eval：不重试，失败切备用（大模型生成.md §6.2）。"""
    primary: str
    fallback: str
    temperature: float
    timeout_s: float


class Scenes(_Model):
    generate: StreamScene
    chat: StreamScene
    preprocess: OneShotScene
    rewrite: OneShotScene
    eval: OneShotScene


class LLM(_Model):
    endpoints: dict[str, Endpoint]
    scenes: Scenes

    @model_validator(mode="after")
    def _scene_refs(self):
        for name in Scenes.model_fields:
            scene = getattr(self.scenes, name)
            for ref in (scene.primary, scene.fallback):
                if ref not in self.endpoints:
                    raise ValueError(f"llm.scenes.{name} 引用了不存在的接入点 {ref!r}")
        return self


# ───────── 后处理 / 评估 / 监控 ─────────

class Notices(_Model):
    unsupported: str
    truncated: str
    interrupted: str


class Postprocess(_Model):
    support_threshold: float
    min_claim_chars: PositiveInt
    max_pairs: PositiveInt
    rerank_max_length: PositiveInt
    verify_timeout_s: float
    warn_ratio: float
    notices: Notices
    incomplete_suffix: str


class EvalGenerate(_Model):
    counts: dict[str, int]
    max_per_doc: PositiveInt
    min_chunk_tokens: int
    min_quote_chars: int
    max_overlap: float
    dedup_similarity: float
    max_regen: int
    unanswerable_max_score: float


class QuickGate(_Model):
    recall_at_8_drop_pp: float
    chat_misroute_max: float
    p95_latency_rise: float


class FullGate(_Model):
    faithfulness_drop: float
    citation_support_drop_pp: float
    unanswerable_drop_pp: float


class Gates(_Model):
    quick: QuickGate
    full: FullGate


class Eval(_Model):
    dataset: str
    baselines_dir: str
    reports_dir: str
    generate: EvalGenerate
    eval_collection_distractor_ratio: float
    ragas_check_samples: PositiveInt
    gates: Gates


class Langfuse(_Model):
    enabled: bool
    host: str
    public_key_env: str
    secret_key_env: str
    mask: bool


class Metrics(_Model):
    path: str


class Feedback(_Model):
    reason_max_chars: PositiveInt
    rate_limit_per_session_min: PositiveInt


class Alert(_Model):
    value: float
    for_: str | None = Field(default=None, alias="for")
    window: str | None = None
    min_count: int | None = None


class Api(_Model):
    docs_list_limit: int = Field(ge=1, le=500)
    request_timeout_s: float = Field(gt=0)


class Monitor(_Model):
    langfuse: Langfuse
    metrics: Metrics
    feedback: Feedback
    alerts: dict[str, Alert]


class Logging(_Model):
    retain_days: PositiveInt
    prompt_sample_rate: float = Field(ge=0, le=1)


class Config(_Model):
    paths: Paths
    services: Services
    ingest: Ingest
    chunking: Chunking
    index: Index
    preprocess: Preprocess
    rewrite: Rewrite
    retrieval: Retrieval
    rerank: Rerank
    expansion: Expansion
    grade: Grade
    context: Context
    prompt: Prompt
    llm: LLM
    postprocess: Postprocess
    eval: Eval
    api: Api
    monitor: Monitor
    logging: Logging


# ───────── 加载 ─────────

class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _no_duplicates(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ConfigError(f"重复的 key {key!r}（第 {key_node.start_mark.line + 1} 行）")
        seen.add(key)
    return loader.construct_mapping(node, deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config(path: str | Path = DEFAULT_PATH, env: Mapping[str, str] | None = None) -> Config:
    """读取并校验配置；`env` 默认取 os.environ，只用于检查 api_key_env 是否存在。"""
    env = os.environ if env is None else env
    try:
        raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML 解析失败：{e}") from e
    try:
        cfg = Config.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(str(e)) from e
    missing = sorted({ep.api_key_env for ep in cfg.llm.endpoints.values() if not env.get(ep.api_key_env)})
    if missing:
        raise ConfigError(f"环境变量未设置：{', '.join(missing)}")
    return cfg


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        load_config(argv[0] if argv else DEFAULT_PATH)
    except ConfigError as e:
        print(f"配置校验失败：{e}", file=sys.stderr)
        return 1
    print("配置校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
