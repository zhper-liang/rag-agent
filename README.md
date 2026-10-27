# ragagent

一个可自建、可替换的 **RAG 问答系统**：离线把 PDF / Word / PPT / 图片 / Markdown 解析、清洗、切分并写入 Milvus，在线用 **LangGraph** 把问答流程建模成有状态图，支持混合检索、重排、引用校验、多轮对话和流式输出。

> 设计目标是「小机器上跑得动、大语料也撑得住」：Embedding 与重排在本地 GPU（RTX 4060 8 GB）上跑，向量库用 Milvus 的 **DISKANN** 磁盘索引，按 1000 万 chunk 设计常驻内存 ≤ 10 GB；LLM 走 OpenAI 兼容接口，主备可切换。

- 全部可调参数集中在 [`config.yaml`](config.yaml)，文档只解释含义，不复制取值。
- 系统设计、接口契约、运维手册见 [`docs/`](docs/README.md)，入口是[总揽](docs/架构与关键信息/总揽.md)。

---

## 目录

- [主要特性](#主要特性)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [离线入库](#离线入库)
- [HTTP 接口](#http-接口)
- [评测](#评测)
- [监控与运维](#监控与运维)
- [文档](#文档)
- [已知限制](#已知限制)
- [许可](#许可)

---

## 主要特性

**离线链路**

- **格式路由**：PDF / Word / PPT / 图片走 MinerU 云 API 解析，Markdown / TXT 本地解析；同一套清洗规则处理页眉页脚、水印、重复图片、重复段落。
- **结构优先切分**：按标题树做层级合并，target / max 双预算，超长代码块按行或语法边界二次切分；标题路径随 chunk 一起入库，既是检索线索也是引用定位依据。
- **图文统一编码**：Visualized-BGE（bge-m3 底座）把文本和图片编到同一个 1024 维空间，一张图和它的说明文字可以互相召回。
- **增量入库**：SQLite 登记表记录文件哈希、版本与解析状态，重跑只处理变化的文档；文档内重复与模板文本标 `searchable=false`，跨文档副本保留。

**在线链路**

- **LangGraph 状态图**：`preprocess → retrieve → rerank → (grade) → build_context → generate → postprocess`，含改写重试与降级分支；闲聊与澄清各有独立路径，所有路径都经过后处理。
- **混合检索**：Milvus 稠密（DISKANN）+ BM25 全文，加权融合后本地 cross-encoder 重排，召回后按 `content_hash` 去重。
- **上下文构造**：6k token 预算内按优先级填充、按文档分组、文档内保持原文顺序，短节整节放入，命中图片时回传原图。
- **引用可追溯**：回答用 `[n]` 标注来源，后处理做格式校验 + 本地重排模型的支持度校验，有问题的引用在界面上标灰；流式正文不改写，只在末尾追加 `final` 事件。
- **可降级**：重排超时、检索降级、LLM 主备切换、整请求超时都会收敛到一次完整回答或明确的错误提示，不会让连接空挂。

**工程化**

- **前端**：Vite + TypeScript + React，流式阶段即渲染 Markdown，收到 `final` 后按 UTF-16 偏移换成引用角标；页内抽屉看原文，支持上传文档与按入库时间筛选。
- **评测**：LLM 生成评测集 + RAGAS 打分，快速 / 完整两档，关键指标不达标返回非零退出码。
- **运维**：systemd 用户单元、live / ready 健康检查、SQLite 在线备份与轮转、Prometheus + Grafana + GPU exporter、Langfuse 云端 trace（只上报遮罩后的元数据）。

---

## 系统架构

```
离线（建库）
  原始文档  PDF / Word / PPT / 图片 ──MinerU 云 API──┐
            Markdown / TXT ──────本地解析──────────┤
                                                    ▼
                          数据清洗 → 文档切分 → 向量化（Visualized-BGE）
                                                    │
                                                    ▼
                        Milvus：DISKANN 稠密索引 + BM25 全文索引 + 标量字段

在线（LangGraph 状态图）
  用户问题 → 问题预处理 → 混合检索 → 重排 → grade ─┬─ 通过 / 降级 → 上下文构造
                │                          │        │
                │                          │        └─ 未通过 → 改写重试（≤2 次）→ 回到检索
                ├─ chat（闲聊，不检索）     └─ 重试耗尽 → 降级回答（列出最接近的来源）
                └─ clarify（澄清）
                                                    ▼
                             Prompt 组装 → 大模型生成 → 后处理 → 回答（SSE 流式）

贯穿：评测（离线题集 + RAGAS）· 监控（Langfuse trace / Prometheus 指标）
```

图定义、State 字段与每条边的条件见[总揽](docs/架构与关键信息/总揽.md) §4。

---

## 技术栈

| 层 | 选型 | 说明 |
|----|------|------|
| 编排 | LangGraph | 有状态图 + SQLite checkpointer，按 `session_id` 隔离会话 |
| 文档解析 | MinerU 云 API | PDF / Word / PPT / 图片 → Markdown + `content_list.json` |
| 向量库 | Milvus 2.5（Standalone） | DISKANN 稠密索引 + 内置 BM25，需要 `queryNode.enableDisk=true` |
| 文档登记 | SQLite（FTS5） | 路径、标题、哈希、版本、解析状态，支撑增量入库与范围识别 |
| Embedding | Visualized-BGE（本地 GPU） | bge-m3 底座，图文同空间，1024 维，fp16 |
| 重排 | bge-reranker-v2-m3（本地 GPU） | cross-encoder，输出 0~1 |
| LLM | OpenAI 兼容接口 | 每个场景（generate / chat / preprocess / rewrite / eval）一个主模型 + 一个备用 |
| API | FastAPI + SSE | 流式回答、原文、图片、上传任务、健康检查、指标 |
| 前端 | Vite + React + TypeScript | 同源部署，构建产物不提交 |
| 评测 | 自研出题器 + RAGAS 0.4.3 | 两档评测 + 回归门槛 + 阈值校准 |
| 监控 | Langfuse + Prometheus + Grafana | trace 只上报元数据；指标与告警在本机 |
| 工具链 | uv / Python 3.12 / pytest / vitest | 依赖精确版本锁定 |

---

## 目录结构

```
ragagent/
├── docs/                     # 三类文档，入口 docs/README.md
│   ├── 架构与关键信息/        # 现行架构、接口、模块设计、运维手册
│   ├── 已完成的计划/          # 历史计划与调优记录（只读）
│   └── 计划/                 # 未完成事项与已知限制
├── src/ragagent/
│   ├── config.py             # config.yaml 的模型定义与启动校验
│   ├── ingest/               # 清洗、MinerU 接入、chunking/ 切分
│   ├── index/                # 编码 / 重排服务、Milvus 建表与写入
│   ├── preprocess/           # 意图路由、指代消解、查询扩展、范围识别、改写
│   ├── retrieval/            # 混合检索、融合、重排、上下文扩展
│   ├── context/              # 上下文构造（预算、分组、引用编号）
│   ├── prompt/               # 模板加载与 Prompt 组装
│   ├── llm/                  # OpenAI 兼容客户端、流式、主备切换
│   ├── postprocess/          # 引用格式与支持度校验、final 事件
│   ├── graph/                # RAGState、nodes/、图构建
│   ├── api/                  # FastAPI 服务、SSE、上传、健康检查
│   └── monitor/              # Langfuse、Prometheus、反馈、备份、GPU exporter
├── web/                      # 前端源码（Vite + React + TS），产物构建到 api/static/
├── prompts/{v1,v2}/          # Jinja2 Prompt 模板，按版本管理
├── eval/                     # 出题、评测、RAGAS、校准、引用格式回归
├── tests/                    # pytest；fixtures/ 契约样例，stubs/ 三个离线桩服务
├── scripts/                  # 冒烟 / 压测 / 备份 / 监控配置生成 / 健康检查
├── deploy/
│   ├── systemd/              # API、编码服务、GPU exporter、备份 timer
│   └── monitoring/           # 预置 Grafana 看板；generated/ 由 config.yaml 生成
├── config.yaml               # 全部可调参数（密钥只写环境变量名）
├── milvus.yaml               # Milvus 服务端配置（DISKANN、mmap）
├── docker-compose.milvus.yml # Milvus standalone（etcd + MinIO + Milvus）
└── pyproject.toml            # 依赖与 pytest 配置（uv 管理）
```

---

## 快速开始

### 1. 环境要求

| 依赖 | 版本 / 说明 |
|------|-------------|
| Python | 3.12（由 [uv](https://docs.astral.sh/uv/) 管理，`.python-version` 已固定） |
| Docker | 运行 Milvus standalone（etcd + MinIO + Milvus 三容器） |
| Node.js | ≥ 22.12（只在构建前端时需要） |
| GPU | NVIDIA 驱动 + CUDA，用于本地编码与重排服务；无 GPU 时可用测试桩替代做联调 |
| 外部服务 | 一个 OpenAI 兼容的 LLM 接入点；MinerU 云 API（解析 PDF / Office / 图片）；Langfuse（可选） |

参考主机：32 核 / 30 GB 内存 / RTX 4060 8 GB。系统按 1000 万 chunk 设计，但**尚未在该规模上实测**（见[已知限制](#已知限制)）。

### 2. 准备模型权重

权重不入库（`models/` 已 gitignore），需要按下面的目录摆放；`src/ragagent/index/gpu_models.py` 启动时会逐个校验这些路径：

```
models/
├── Visualized_m3.pth                      # HuggingFace: BAAI/bge-visualized
├── BAAI/
│   ├── bge-m3/                            # config、tokenizer（文本底座）
│   └── bge-reranker-v2-m3/                # 重排模型权重与 tokenizer
└── FlagEmbedding/
    └── research/visual_bge/               # FlagEmbedding 仓库的 research/visual_bge
```

- 权重路径可覆盖：`index.embed_weight` 在 `config.yaml`；另外三项可用编码服务的 `--visual-code` / `--text-model` / `--rerank-model` 参数指定。
- 缺少任一文件时编码服务启动即报错，不会静默降级。

### 3. 安装依赖与配置

```bash
uv sync                                   # 安装 Python 依赖（精确版本见 uv.lock）
cp .env.example .env                      # 然后填写真实密钥；.env 已 gitignore，不要提交
```

`.env` 里只有密钥，地址、模型名、端口等都在 `config.yaml`：

| 变量 | 用途 |
|------|------|
| `MILVUS_USER` / `MILVUS_PASSWORD` | Milvus 账号鉴权 |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | Milvus 的对象存储凭据；**已有 `volumes/` 数据时，改凭据前要先清空 `volumes/minio`** |
| `MINERU_TOKEN` | MinerU 云 API 令牌（解析 PDF / Office / 图片） |
| `LLM_MAIN_KEY` / `LLM_MAIN_BACKUP_KEY` | 生成场景的主 / 备接入点密钥 |
| `LLM_SMALL_KEY` / `LLM_SMALL_BACKUP_KEY` | 预处理、改写、闲聊场景的主 / 备接入点密钥 |
| `GRAFANA_ADMIN_PASSWORD` | Grafana 管理员密码（监控栈可选） |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse 云端 trace（可选） |

### 4. 启动（顺序不能颠倒）

```
Milvus（compose healthy） → 编码 + 重排服务（/health 200） → 问答 API
```

**① Milvus**

```bash
docker compose -f docker-compose.milvus.yml up -d
docker compose -f docker-compose.milvus.yml ps        # 等到 healthy
```

compose 文件不设 `restart` 策略，容器生命周期由使用者自己管理。停止用 `down`。

**② 编码 + 重排服务（GPU，端口见 `services.embed`）**

```bash
uv run --env-file .env python -m ragagent.index.embed_service
```

模型加载需要几十秒，期间 `/health` 返回 503。加载完成后：

```bash
curl -s http://127.0.0.1:8001/health
```

**③ 问答 API + 网页**

```bash
uv run --env-file .env uvicorn ragagent.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

- **只能跑 1 个 worker**，不要加 `--workers`：会话串行锁、LLM 并发上限和 Prometheus 指标都在进程内。
- 没有登录鉴权，只绑 `127.0.0.1`；要从别的机器访问请用 SSH 端口转发，不要改成 `0.0.0.0`。
- 打开 <http://127.0.0.1:8000/>。

### 5. 构建前端

前端源码在 `web/`，构建产物输出到 `src/ragagent/api/static/`，**不提交**。首次部署和前端有改动后：

```bash
cd web
npm ci
npm run build          # 内含 tsc --noEmit + vite build
cd ..
```

没有构建产物时 API 照常工作，但 `/`、`/c/*`、`/d/*` 返回 503 和「前端未构建」提示。前端开发调试用 `npm run dev`（默认 5173，已把 `/api` 代理到 8000）。

### 6. 验收

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/api/health/live
curl --fail --silent --show-error http://127.0.0.1:8000/api/health/ready
uv run --env-file .env python scripts/check_health.py
```

- `live`：进程能处理请求就返回 200，不探测外部依赖。
- `ready`：并发检查 Milvus 别名可查询、编码服务 `/health`、登记表可读；全成功才 200，否则 503 并在 `checks.milvus` / `checks.embed` / `checks.registry` 里给出原因类型（不返回异常原文，也不探测 LLM）。

### 7. 开机自启（可选，systemd 用户单元）

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/*.service deploy/systemd/*.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now ragagent-embed ragagent-api ragagent-gpu
systemctl --user enable --now ragagent-backup.timer
loginctl enable-linger "$USER"          # 不登录也开机启动
```

- 单元文件里的 `%h/code/ragagent`（仓库路径）和 `%h/.local/bin/uv`（uv 路径）按实际路径修改。
- Milvus 不在 systemd 管理范围内，仍需手动 `docker compose up -d`。
- 状态与日志：`systemctl --user status ragagent-api`、`journalctl --user -u ragagent-api -f`。
- 改代码或 `config.yaml` 后重启 API 即可；尽量别重启 `ragagent-embed`（重新加载模型要几十秒）。

---

## 离线入库

三段式：**清洗 → 切分 → 建库**。入库命令手动执行；写入前会对 `paths.ingest_lock` 取 `flock`，拿不到立即非零退出，进程退出自动释放。

```bash
# ① 清洗：解析、去噪、去重、登记（--local-only 只处理本地 md/txt，不上传云端）
uv run --env-file .env python -m ragagent.ingest.pipeline <input_dir> [--limit N] [--local-only]

# ② 切分：单独调试时用（入库流程内部也会跑）
uv run --env-file .env python -m ragagent.ingest.chunking.pipeline <input> [--output <dir>]

# ③ 向量化并写入 Milvus（需要 Milvus 与编码服务已就绪）
uv run --env-file .env python -m ragagent.index.pipeline <chunk_dir> [--force] [--no-wait]
```

- 入库与在线问答共用同一块 GPU 和编码服务进程，大批量入库期间在线检索延迟会上升，**建议放在低峰跑**。
- 也可以直接在网页上传文档（`POST /api/uploads`），后台串行执行完整入库链路；上传根 `paths.uploads` 与 `paths.raw` 相互独立。
- Milvus 冒烟与内存 / 召回实测：`uv run --env-file .env python scripts/milvus_smoke.py`、`scripts/milvus_bench.py {mmap,recall}`（各自使用独立集合，不动正式数据）。

---

## HTTP 接口

| 方法 | 路径 | 用途 |
|------|------|------|
| `POST` | `/api/chat` | 问答入口，返回 SSE 流。body：`question`、可选 `session_id` / `filters` / `time_from` / `time_to` / `use_scope`；也可用 `X-Session-ID` 头，二者同传必须一致 |
| `GET` | `/api/docs` | 已入库文档列表（`?q=` 按标题匹配），不返回本地路径 |
| `GET` | `/api/docs/{doc_id}/file` | 取该文档清洗后的 `full.md`；默认下载，`?inline=true` 以纯文本打开 |
| `GET` | `/api/images/{doc_id}/v{version}/{filename}` | 文档内图片 |
| `POST` | `/api/uploads` | 上传文档（multipart，返回 `job_id`，异步入库） |
| `GET` | `/api/uploads` · `/api/uploads/{job_id}` | 上传任务列表 / 单个任务进度 |
| `POST` | `/api/uploads/{job_id}/retry` | 重试被中断的上传任务 |
| `POST` | `/api/feedback` | 点赞 / 点踩（`rating` 为 1 或 -1，绑定本会话的 `message_id`） |
| `GET` | `/api/health/live` · `/api/health/ready` | 存活 / 就绪检查 |
| `GET` | `/metrics` | Prometheus 指标 |
| `GET` | `/`、`/c/{session_id}`、`/d/{doc_id}` | 前端页面（History API 路由，需先构建前端） |

**SSE 事件**只有五种，均为 `data: {json}\n\n` 帧：

| `type` | 含义 |
|--------|------|
| `progress` | 阶段推进（`preprocessing` / `retrieving` / `reranking` / `rewriting` / `building_context` / `generating`） |
| `token` | 增量正文 |
| `meta` | 生效范围、附件图片等元信息 |
| `final` | 回答结束：引用偏移（UTF-16）、来源列表、支持度标记 |
| `error` | 出错或被取消（整请求超时也会补发一个 `error` 收尾） |

上传接口是写操作，额外要求自定义请求头 `X-RAG-Client` 并校验 `Origin` / `Host` 同源——这不是身份认证，只挡跨站表单与 DNS rebinding。**所有接口都没有鉴权，只应在本机或可信内网使用。**

---

## 评测

评测集由 LLM 生成（标准答案记「文档 + 原文片段」，不绑定 chunk_id），RAGAS 打分，分快速 / 完整两档：

```bash
make test                                        # 后端测试（uv run pytest）
make test-web                                    # 前端测试（需先在 web/ 下 npm ci）
uv run python -m eval.run --tier quick --fixture  # 离线自检，不调用任何外部服务
```

真实评测需要显式预算，避免误触发大量计费调用：

```bash
uv run --env-file .env python -m eval.run --tier quick \
  --chunks current.jsonl --dataset eval/datasets/v1.jsonl --max-calls N
```

- 退出码：`0` 全部门槛通过；`1` 回归失败 / 缺指标 / 基线不兼容 / 波动超限；`2` 输入或后端错误。
- `--fixture` 只验证评测壳，数值一律标记 fixture，不会创建生产基线。
- 报告与数据集含资料原文片段，写入 gitignore 的 `eval/datasets/`、`eval/reports/`，不要提交。
- 详细口径（出题、RAGAS 中文对照、阈值校准、引用格式回归）见 [`eval/README.md`](eval/README.md)。

当前状态：471 题评测集已建成；正式基线是 2026-10-19 的 **40 题分层样本双跑**，不是 471 题全量结果；后续调优轮次只在固定小题集上做诊断，未回写基线。历次数字见[项目开发与调优日志](docs/已完成的计划/项目开发与调优日志.md)。

---

## 监控与运维

**Prometheus + Grafana + GPU exporter**（配置全部由 `config.yaml` 的 `monitor.stack` 生成，不要手改生成文件）：

```bash
uv run --env-file .env python scripts/gen_monitoring.py
docker compose --env-file .env -p ragagent-monitoring -f deploy/monitoring/generated/compose.yml up -d
```

- Grafana 管理密码只通过 `GRAFANA_ADMIN_PASSWORD` 注入；Prometheus 9090、Grafana 3000 只发布到主机回环。
- 看板覆盖阶段延迟 p50/p95/p99、错误率、降级率、GPU 显存、Milvus 常驻内存；告警规则在 Prometheus 内评估，**未接外部通知渠道**。
- 本机 GPU exporter 由 `ragagent-gpu.service` 提供（`nvidia-smi`，不需要容器 GPU runtime）。

**备份**：`scripts/backup.py` 对登记表、反馈库、会话库做 SQLite 在线备份并轮转（`ragagent-backup.timer` 每日触发，错过会补跑）。Milvus 不打包备份——保留原始 / 清洗 / chunk 数据与配置，用新集合重建后再切别名。

```bash
uv run --env-file .env python scripts/backup.py
systemctl --user list-timers ragagent-backup.timer
```

**Langfuse**：云端 trace 只上报元数据，字符串字段走白名单 + 掩码，资料原文、用户问题与点踩理由都不上传。

部署细节、健康语义、备份恢复步骤见[服务化部署](docs/架构与关键信息/服务化部署.md)。

---

## 文档

`docs/` 分三类，入口是 [`docs/README.md`](docs/README.md)：

| 类别 | 内容 |
|------|------|
| [架构与关键信息](docs/架构与关键信息/) | 现行架构、接口契约、模块设计与运维手册（改代码要同步） |
| [已完成的计划](docs/已完成的计划/) | 历史计划与调优记录，只读 |
| [计划](docs/计划/后续工作.md) | 未完成事项、已知限制、下一步 |

建议阅读顺序：

1. [总揽](docs/架构与关键信息/总揽.md) —— 全景、LangGraph 图与 State、技术栈、目录树、决策记录、验证边界
2. 离线：[数据清洗](docs/架构与关键信息/数据清洗.md) → [文档切分](docs/架构与关键信息/文档切分.md) → [向量化建库](docs/架构与关键信息/向量化建库.md)
3. 在线：[用户问题预处理](docs/架构与关键信息/用户问题预处理.md) → [检索与重排](docs/架构与关键信息/检索与重排.md) → [上下文构造](docs/架构与关键信息/上下文构造.md) → [Prompt 组装](docs/架构与关键信息/Prompt%20组装.md) → [大模型生成](docs/架构与关键信息/大模型生成.md) → [后处理](docs/架构与关键信息/后处理.md)
4. 贯穿与运维：[评估与监控](docs/架构与关键信息/评估与监控.md)、[服务化部署](docs/架构与关键信息/服务化部署.md)、[启动方法](docs/架构与关键信息/启动方法.md)

跨模块的数据结构以 pydantic v2 模型为准，文档中的定义与各包 `models.py` 逐字一致，并由 `tests/test_contracts.py` 按 AST 比对——改字段必须同时改模型、文档、调用方与夹具。开发约定见 [`CLAUDE.md`](CLAUDE.md)。

---



## 许可

本项目以 [MIT License](LICENSE) 开源，Copyright (c) 2026 zhper-liang。

第三方组件遵循各自许可证：Milvus（Apache-2.0）、LangGraph、FastAPI、RAGAS、Visualized-BGE / bge-m3 / bge-reranker-v2-m3（BAAI 模型许可）、MinerU 云服务条款。模型权重与真实语料不随本仓库分发。
