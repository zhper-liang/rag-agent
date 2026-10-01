# L4 评测与监控使用说明

## 边界

- `--fixture` **仅为合成壳验证**：消费 P0 夹具，所有生成/评分数值都标记 fixture；
  基线独立存放 `eval/baselines/fixtures/`，不会创建生产基线。不能用于 M4 或阈值决策。
- 真实评测直接调用 L2 quick / L3 graph，不挂 Langfuse，关闭 RAGAS 遥测。
  Milvus/登记表只读，临时集合由 L1 构建并使用 `l4_` 前缀。本线不使用 GPU。
- `.env` 只通过 `uv run --env-file .env` 加载，不输出密钥。D-002 已允许免费额度 API；
  批量前仍应确认接入点的免费额度及本批次上限。下面命令中的 `N` 必须替换为本次确认的调用数。
- `--max-calls` 限制出题/独立 RAGAS 评测模型请求；`eval.run --max-eval-calls` 仅限制评分/判定，
  不包含 eval.run 的 L2 预处理和 L3 生成链调用。它们按问题数及主备/重试配置另行估算额度。
  出题器的无答案过滤中预处理也计入该出题批次的 `--max-calls`。
- 本地报告/状态导出包含真实问题、上下文和答案，不能提交/复制到测试或上传到 Langfuse。

## 快速自检（无任何外部服务调用）

```bash
uv run python -m eval.run --tier quick --fixture
uv run python -m eval.run --tier full --fixture
# 明确更新合成基线，仅供壳验证
uv run python -m eval.run --tier full --fixture --update-baseline
# 应返回1，基线不被覆盖
uv run python -m eval.run --tier quick --fixture --fixture-regression
uv run python -m eval.run --tier full --fixture --fixture-regression
```

退出码：`0` 所需门槛全部通过；`1` 回归失败/缺失指标/缺失或不兼容基线/波动超限；
`2` 输入或后端执行错误。不自动建立生产基线。百分点门槛除以100，刚好等于门槛允许。
首次完整档 `--update-baseline` 跑两次，对称比较指标，任一方向超过配置门槛均不写基线。
报告 JSON 和 Markdown 写入 `eval.reports_dir`，失败标红；fixture 报告在其 `fixtures/` 子目录。

## 真实出题与评测（M1b 完成后）

准备当前 chunk JSONL（chunk模型字段），真实本地编码/重排服务和索引、登记表。

```bash
uv run --env-file .env python -m eval.generate --chunks current.jsonl --max-calls N \
  --output eval/datasets/v1.jsonl
uv run --env-file .env python -m eval.run --tier quick --chunks current.jsonl \
  --dataset eval/datasets/v1.jsonl
uv run --env-file .env python -m eval.run --tier full --chunks current.jsonl \
  --dataset eval/datasets/v1.jsonl --max-eval-calls N \
  --states-output eval/reports/actual-samples.jsonl
# 经检查后显式加 --update-baseline；完整档首次需要两倍问题/评分预算
```

`eval.run` 默认从 `cfg.paths.chunks` 递归读 JSONL；未完成 M1b 时会明确失败，绝不回退 fixture。
内置依赖：`eval.runtime.open_deps` 用 L1 TokenCounter、只读 RegistryDB、MilvusClient 构造 L3 Deps；
目标 collection 默认 `services.milvus.collection`（线上别名 `rag_chunks`），可 `--collection l4_...`。
可用 `--deps-factory module:function` 自定义 `(cfg,collection)->Deps`（支持async）。外部工厂负责自己创建的
HTTP/其他资源；默认工厂会关闭登记表/Milvus/LLM客户端。不要在工厂中启用 Langfuse。

出题器对 `unanswerable` 用当前 L2 检索最高重排分过滤，默认使用内置只读依赖，也可注入 deps factory。
工厂最小示例（`eval_local.py` 放在本地，不提交密钥/真实数据）：

```python
import os
from pymilvus import MilvusClient
from ragagent.graph.deps import Deps
from ragagent.ingest.chunking.tokenizer import get_counter
from ragagent.ingest.registry_db import RegistryDB

def create_deps(cfg, collection):
    svc = cfg.services.milvus
    # L4传入的cfg已指向所选collection；本工厂也应显式维护此契约。
    cfg = cfg.model_copy(update={'services': cfg.services.model_copy(update={
        'milvus': svc.model_copy(update={'collection': collection})})})
    return Deps(cfg=cfg, count=get_counter(cfg.chunking.tokenizer),
                milvus=MilvusClient(uri=svc.uri, user=os.environ.get(svc.user_env, ''),
                                    password=os.environ.get(svc.password_env, '')),
                registry=RegistryDB(cfg.paths.registry_db, readonly=True))
```

出题数量/文档上限/最短chunk与quote/照抄过滤/向量去重/重出次数全部读 `eval.generate`。
只读取当前版本可检索chunk，按文档轮转抽样；失败及缺题写 `.generation.json`，不足返回1。
`gold` 始终是 doc_id/version/quote，不绑定chunk_id。多轮前轮有answer，末轮有standalone；
跨段必须分别有两个chunk的quote；image单段必须带image标签；keyword由出题prompt识别精确词。

## RAGAS 与中文对照

固定0.4.3（legacy包装器接口按设计使用，SDK会提示弃用；升级前须重做兼容性验收）。
五项真实RAGAS指标：忠实度、回答相关性、上下文精确率/召回率、正确性。
回答复用 L3 parser 去补充段和非代码 `[n]`，无答案题不跑RAGAS，只做fallback/直接判定。
回答相关性/语义相似度均使用本地 `/embed/text {texts:[...]}`，禁止使用默认远程embedding。
主备采用 L3 eval scene/客户端/并发限制；SDK/RAGAS隐式重试关闭，所有评分请求有调用预算。

```bash
uv run --env-file .env python -m eval.ragas_eval \
  --input eval/reports/actual-samples.jsonl --output eval/reports/chinese-check.json \
  --check-chinese --max-calls N
```

输入每行 `{case:完整评测题,state:实际RAGState}`，需要30条互异、有gold的中文题。
逐条与同标准直接LLM评分对照，输出每项 MAE；默认人工审查阈值0.15，可 `--max-difference`。
超限返回1，要求人工检查/中文prompt适配后再验证，不自动宣称中文效果合格。
适配后的RAGAS中文prompt JSON文件可通过 `--prompt-dir`/`eval.run --ragas-prompt-dir` 加载。
本地HTTP合成桩验证的是SDK链路与计算，不是中文评分质量。

## 校准（仅建议，不写配置）

```bash
uv run python -m eval.calibrate --input observations.json --mode real --output eval/reports/calibration.json
```

JSON输入格式（每节都是实测记录列表；空节保持未完成）：

| 键 | 每条记录 |
|---|---|
| `grade_samples` | `{score:重排最高分,label:有答案bool}` |
| `support_pairs` | `{score:句子引用重排分,label:直接评测支持bool}` |
| `rerank_trials` | `{threshold,candidates:[{chunk_id,score}],gold:[[当前对齐id,...],...]}` |
| `weight_trials` | `{dense_weight,ranked:[id,...],gold:[[id,...]],tags:[keyword或image]}` |
| `retrieval_trials` | `{candidates,search_list,ranked,gold,latency_s}` |
| `token_samples` | `{chunk_id,tokens,prompt_tokens,fixed_overhead}` |
| `chunk_trials` | `{target_tokens,ranked,gold,latency_s}` |

另提供 `provenance`（数据集版本/索引collection/模型/配置hash/实测日志），供人工复核；
`--mode real` 只是声明输入来源，不能替代独立的真实性审计。
- grade 用 Youden J；支撑阈值用 F1，含混淆矩阵及完整曲线。
- 重排丢失按gold片段计数，原本没召回单独列，不归咎于阈值。
- 稠密权重0.50~0.80每0.05步，输入必须是每次实际检索排名，不近似模拟Milvus归一化。
- candidates/search_list 给出 Recall@8/p95 延迟及 Pareto 前沿。
- token系数减固定开销，1000个不同chunk的p95比值；样本不足不推荐。
- chunk目标256/400/600，缺任何实验不推荐。
- `--mode synthetic` 可以测逻辑，但报告永远 `real_calibration_complete=false`。

## 在线监控

交付消息：metrics/langfuse #026；反馈router #027。
- `MonitorMetrics.router()` 暴露 `/metrics`，标签白名单固定；对象可注入registry供隔离测试。
- Langfuse创建client时强制mask，并对异常回调移除原文；上报只有元数据，reason不上传。
- `create_feedback_router` 必须注入与问答相同的session resolver和checkpointer历史查询。
  SQLite保存rating/reason，事务限流；本地点踩由`export_downvotes`导出。

```bash
uv run --env-file .env python scripts/gen_alerts.py --output eval/reports/alerts.yml
# 有promtool时进一步校验PromQL；本地未安装时已有YAML结构校验
promtool check rules eval/reports/alerts.yml
uv run pytest tests/eval tests/monitor
make test
```

告警阈值/窗口/持续时间/min_count只读 `monitor.alerts`。外部Milvus集成测试默认跳过，
`make test` 中的skip不能当作该外部项通过；真实M4缺项见协作待决单。
