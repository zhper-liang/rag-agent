# L4 评测与监控使用说明

## 边界

- `--fixture` **仅为合成壳验证**：消费 P0 夹具，所有生成/评分数值都标记 fixture；
  基线独立存放 `eval/baselines/fixtures/`，不会创建生产基线。不能用于 M4 或阈值决策。
- 真实评测直接调用 L2 quick / L3 graph，不挂 Langfuse，关闭 RAGAS 遥测。
  生产 Milvus/登记表只读；校准通过 L1 已交付接口构建 `l4_` 临时集合。
  本线消费 8001 HTTP 编码/重排服务，不启动 GPU。
- `.env` 只通过 `uv run --env-file .env` 加载，不输出密钥。D-008~D-012 已授权真实批量运行和两条 masked 合成云端冒烟。
  下面命令中的 `N` 必须替换为本批次显式调用上限。
- 所有真实 CLI 必须显式 `--max-calls`；发送 HTTP 请求前按实际次数限制，包含主备、重试、
  L2 预处理、L3 生成及 judge。`eval.run --max-eval-calls` 是 judge 的额外保守预约上限。
  `.calls.json` 记录实际发送次数、状态/异常类和 usage，不包含文本或密钥；embed/rerank 不计 LLM 请求。
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
`--concurrency` 默认为1，需显式设置相同并发以比较两跑；结果按题集顺序保存，失败题不剔除。
首次完整档 `--update-baseline` 跑两次，对称比较指标，任一方向超过配置门槛均不写基线。
报告 JSON 和 Markdown 写入 `eval.reports_dir`，失败标红；fixture 报告在其 `fixtures/` 子目录。

## 真实出题与评测（M1b 完成后）

准备当前 chunk JSONL（chunk模型字段），真实本地编码/重排服务和索引、登记表。

```bash
uv run --env-file .env python -m eval.generate --chunks current.jsonl --max-calls N \
  --output eval/datasets/v1.jsonl
uv run --env-file .env python -m eval.run --tier quick --chunks current.jsonl \
  --dataset eval/datasets/v1.jsonl --max-calls N
uv run --env-file .env python -m eval.run --tier full --chunks current.jsonl \
  --dataset eval/datasets/v1.jsonl --max-calls N --max-eval-calls N \
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
`--concurrency` 默认为1；并发生成按题型轮转并预留文档名额，保证去重和每文档上限。
每个接受题即时落盘，可用 `--resume-from` 从已有 JSONL 续跑（输出使用新文件）；恢复继承名额、
题型计数和去重向量，必须合并原批与恢复批 `.calls.json` 的实际次数。无答案题的恢复需
`gen.source_doc_ids`；题号沿用。后端异常、缺题和中断保留，不自动补成成功。

`gold` 始终是 doc_id/version/quote，不绑定chunk_id。多轮前轮有answer，末轮有standalone；
跨段必须分别有两个chunk的quote；image单段必须带image标签；keyword由出题prompt识别精确词。

## RAGAS 与中文对照

固定0.4.3（legacy包装器接口按设计使用，SDK会提示弃用；升级前须重做兼容性验收）。
五项真实RAGAS指标：忠实度、回答相关性、上下文精确率/召回率、正确性。
回答复用 L3 parser 去补充段和非代码 `[n]`，无答案题不跑RAGAS，只做fallback/直接判定。
回答相关性/语义相似度均使用本地 `/embed/text {texts:[...]}`，禁止使用默认远程embedding。
主备采用 L3 eval scene/客户端/并发限制；SDK/RAGAS隐式重试关闭，所有评分请求有调用预算。单个接入点请求遵守eval scene timeout；
RAGAS metric总时限另按上下文数量及主备串行请求上界分配，避免把60s请求时限误当整项metric时限。RAGAS多采样拆成独立 `n=1` 实际请求，
兼容不支持 `n>1` 的接入点；每个样本的主备请求逐次计数，不复制回答冒充独立采样。

```bash
uv run --env-file .env python -m eval.ragas_eval \
  --input eval/reports/actual-samples.jsonl --output eval/reports/chinese-check.json \
  --check-chinese --max-calls N
```

输入每行 `{case:完整评测题,state:实际RAGState}`，需要30条互异、有gold的中文题。
`--concurrency` 控制对照样本并发；每题结束即写进度检查点，最终报告包含失败和missing。
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

真实采集入口（每阶段独立文件，显式预算；生产索引不改）：

```bash
uv run --env-file .env python -m eval.collect_calibration --phase tokens \
  --chunks current.jsonl --output tokens.json --max-calls 1001
uv run --env-file .env python -m eval.collect_calibration --phase retrieval \
  --chunks current.jsonl --dataset cases.jsonl --output retrieval.json --max-calls 1
uv run --env-file .env python -m eval.collect_calibration --phase chunks \
  --chunks current.jsonl --dataset cases.jsonl --output chunks.json --max-calls 1 \
  --collection-prefix l4_calibration_unique
uv run --env-file .env python -m eval.collect_calibration --phase support \
  --chunks current.jsonl --states actual-samples.jsonl --output support.json --max-calls N
```

切分实验仅重切抽样问题涉及文档及5%干扰文档，256/400/600正式接口切分、编码、入库后真实检索。
已存在的集合拒绝覆盖；报告须写明样本数、文档数和 stale gold。检索阶段用末轮/standalone问题，
不调用预处理 LLM。支撑阶段只消费真实答案的句子-引用对，不制造负例；缺任一类别标记未完成。
采集阶段分别保留失败与 ledger，合并观测后运行校准分析；建议不自动写配置。

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


## 引用格式离线回归

只消费已保存的双跑 `states.jsonl`，比较旧 `[n]` 规则和兼容 `[n]` / `【n】` 规则；不调用生成、重排或评测模型，不重写输入、不更新基线。`--heading` 应取 `config.yaml` 的 `prompt.supplement_heading`，例如当前项目可运行：

```bash
uv run python -m eval.citation_format_regression --states .planning/L4/m4/m4final5-states.jsonl --heading "$(uv run python -c 'import yaml; print(yaml.safe_load(open("config.yaml"))["prompt"]["supplement_heading"])')" --output eval/reports/citation-format-regression.json
```

历史原件只读，报告写入 gitignored `eval/reports/`。`added_unverified` 只表示编号存在，不能当作受支持；支持率仍需在原阈值下对真实句子-引用窗口重新打分。本回归显式记录 `support_revalidated=false`。

### 2026-10-25 评分校准约定
事实参考`reference`只存答案事实，必答/可选/禁答规则存独立rubric；单轮评分使用当前问题，多轮使用与当前对话一致的standalone。明确指定文档的题目必须标注scope_doc_ids。保存states包括soft_filters、retrieval、hits、candidates和standalone_question，支持逐题范围与筛选诊断。旧报告不可覆盖；评分标准变更后的重评单列，不记为系统能力收益。
