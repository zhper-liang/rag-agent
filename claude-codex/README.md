# 四线并行协作协议

四个 Claude Code 窗口（L1~L4）在**同一个工作目录、同一个 `main` 分支**上并行实现 `docs/开发计划.md` §4 的四条线。窗口之间只通过本目录的文件交流。用户是唯一的裁决人，并接替 P0 负责人管共享文件。

每个窗口开工前读完本文件和自己的任务书（`L1.md` ~ `L4.md`）。本文件与 `CLAUDE.md` 冲突时以 `CLAUDE.md` 为准。

## 1. 启动（用户）

1. 启动 Docker Desktop，再起 Milvus：`docker compose -f docker-compose.milvus.yml up -d`。L1、L2、L4 要用它；不起也能开工，相关测试会跳过。
2. 开四个终端，都在仓库根目录：

   ```
   PLAN_ID=L1 claude -n L1
   PLAN_ID=L2 claude -n L2
   PLAN_ID=L3 claude -n L3
   PLAN_ID=L4 claude -n L4
   ```

   `PLAN_ID` 让 planning-with-files-zh 的钩子读 `.planning/L1/` 这类目录。漏了它，钩子会把别的线的计划注入进来。

3. 每个窗口发一条引导词，二选一（把 L1 换成对应的线）：

   普通模式：

   ```
   你是 L1 线。先完整读 claude-codex/README.md 和 claude-codex/L1.md，然后按 L1.md 一直做到「完成标准」全部满足。用 planning-with-files-zh 技能记录进度，规划文件放在 .planning/L1/。
   ```

   目标模式：

   ```
   /goal 你是 L1 线：读完 claude-codex/README.md 和 claude-codex/L1.md，按 L1.md 执行，直到「完成标准」每一条都满足并贴出验证输出；剩下做不了的每一项都已写进 claude-codex/待决.md 并在等用户裁决。进度用 planning-with-files-zh 记在 .planning/L1/
   ```

4. 用户日常要做的事：
   - 看 `状态.md`，了解每条线做到哪、卡在哪。
   - 在 `待决.md` 的条目下面写一行 `**用户裁决：** …`。各线会定期读；急的话在对应窗口说一声。
   - 给某条线留言：`claude-codex/board.sh post 消息 user L2 通知 "标题" <<'EOF'`（正文写在 heredoc 里）。

## 2. 本目录的文件

| 文件 | 用途 | 谁写 |
|------|------|------|
| `README.md` | 本协议 | 用户 |
| `L1.md` ~ `L4.md` | 各线任务书 | 用户 |
| `状态.md` | 各线当前状态 | 各线只改自己那一节（先取锁 `status`） |
| `消息.md` | 线与线之间的消息，只追加 | 只用 `board.sh post 消息 …` 写 |
| `待决.md` | 要用户拍板的事 | 各线用 `board.sh post 待决 …` 提出；用户在条目下写裁决 |
| `board.sh` | 锁和发消息 | 不改 |
| `.locks/` | 锁目录 | 只由 `board.sh` 管 |

`状态.md`、`消息.md`、`待决.md`、`.locks/` 和 `.planning/` 已加进 `.gitignore`，不提交。

## 3. 目录归属

一条线只改自己名下的文件。要改别人的文件，就给对方发「请求」，由对方来改。

| 线 | 独占的代码 | 独占的测试 |
|----|-----------|-----------|
| L1 离线入库 | `src/ragagent/ingest/`（含 `chunking/`）、`src/ragagent/index/` 下除 `models.py`、`schema.py` 外的文件 | `tests/ingest/`、`tests/chunking/`、`tests/index/` |
| L2 检索、上下文、预处理 | `src/ragagent/retrieval/`、`src/ragagent/context/`、`src/ragagent/preprocess/`；`graph/nodes/` 下的 `retrieve.py`、`rerank.py`、`build_context.py`、`routes.py`、`preprocess.py`、`clarify.py`、`rewrite.py` | `tests/retrieval/`、`tests/context/`、`tests/preprocess/` |
| L3 生成链与组装 | `src/ragagent/llm/`、`src/ragagent/prompt/`、`src/ragagent/postprocess/`、`src/ragagent/api/`；`graph/build.py`、`graph/deps.py`；`graph/nodes/` 下的 `generate.py`、`postprocess.py`、`fallback.py` | `tests/llm/`、`tests/prompt/`、`tests/postprocess/`、`tests/api/`、`tests/integration/` |
| L4 评测与监控 | `eval/`、`src/ragagent/monitor/`、`scripts/gen_alerts.py` | `tests/eval/`、`tests/monitor/` |

- 每个测试目录都要有 `__init__.py`，否则同名测试文件会冲突。
- `graph/deps.py` 归 L3。L2 可以在 `graph-deps` 锁里追加自己节点要用的字段，但不能改已有字段，改完通知 L3。
- 开发计划 §4 没有分配预处理、图组装、fallback 节点和 API 主程序。这里先分给 L2 和 L3，见 `待决.md` D-004，用户可以改判。
- 各包里已有的 `models.py`、`schema.py`、`preprocess/schema.py` 是 P0 契约。它们要和文档逐字一致，属于共享文件。

**共享文件**（任何一条线都不能直接改，按 §6 走）：

- `config.yaml`、`src/ragagent/config.py`、`tests/test_config.py`
- `src/ragagent/graph/state.py`；各包已有的 `models.py` / `schema.py`
- `tests/fixtures/` 里已有的文件、`tests/conftest.py`、`tests/test_*.py`（P0 测试）、`tests/stubs/`
- `docs/`、`CLAUDE.md`、`prompts/`
- `pyproject.toml`、`uv.lock`
- `docker-compose.milvus.yml`、`milvus.yaml`、`.env.example`、`.gitignore`、`Makefile`、`scripts/milvus_*.py`

## 4. 锁

锁就是 `.locks/<锁名>/` 目录（`mkdir` 是原子操作）。持有者写在 `owner` 文件里。

```
claude-codex/board.sh run <锁名> <线> -- <命令...>   # 持锁执行，结束自动释放（首选）
claude-codex/board.sh lock <锁名> <线>               # 取锁；之后用 Edit 改文件
claude-codex/board.sh unlock <锁名> <线>             # 改完立刻释放
claude-codex/board.sh locks                          # 查看谁持有什么
```

| 锁名 | 保护什么 |
|------|----------|
| `git` | 所有 `git add` / `git commit`（git 的暂存区只有一个） |
| `deps` | `uv add` / `uv remove`、`pyproject.toml`、`uv.lock` |
| `config` | `config.yaml`、`config.py`、`tests/test_config.py` |
| `docs` | `docs/`、`CLAUDE.md`、`prompts/` |
| `fixtures` | `tests/fixtures/`、`tests/conftest.py`、`tests/test_*.py`、`tests/stubs/` |
| `graph-deps` | `src/ragagent/graph/deps.py` |
| `status` | `状态.md` |
| `board-msg` / `board-decision` | `board.sh post` 自动使用，不要手动取 |

- 锁只在改文件或执行命令的那一小段时间里拿着，不要拿着锁去跑长任务或等别人。
- 等锁超过 10 分钟，`board.sh` 会报错退出，并打印持有者。先给持有者发消息。如果持有者的窗口已经不在了，请用户删掉锁目录，不要自己删别人的锁。

## 5. 消息

**发送**：只能用 `board.sh post`。它会自动编号、取锁、追加，并打印编号。

```
claude-codex/board.sh post 消息 L3 L2 交付 "structured_call 可用" <<'EOF'
模块：src/ragagent/llm/structured.py
签名：async def structured_call(scene, eps, schema, messages)（大模型生成 §6.2）
提交：abc1234
注意：返回 None 的处理见 D-005
EOF
```

- 收件：`L1`~`L4`、`user` 或 `全体`。
- 类型：`通知`（告知变化）、`请求`（要对方做事）、`回复`（标题写 `回复 #012 …`）、`交付`（接口可用了）、`阻塞`（被别人卡住）。
- 正文要能独立看懂：写清模块路径、函数签名、提交号、文档章节。不写「如上」「同前」。
- 已发的消息不改。要更正就再发一条。

**接收**：`claude-codex/board.sh inbox L2 <上次读到的编号>`，列出发给你或发给全体的新消息。读到的最大编号记进 `.planning/<线>/progress.md`。

要查收件箱的时机：开工时、每个阶段开始和结束时、提交前、被卡住时。

收到「请求」后，在下一个检查点之前回复，至少回一句「收到，排在第几步」。

## 6. 共享文件与设计改动

改动分两类。

**A 类：只影响本线内部**，例如模块内部结构、私有函数、本线测试数据。自己决定，记进 `.planning/<线>/findings.md`。

**B 类：其他所有改动**，包括文档里的设计、跨模块契约（`models.py`、`state.py`、已有夹具）、`config.yaml` 已有的 key、`prompts/`，以及和文档签名不一致的跨线接口。

1. 用 `board.sh post 待决 <线> user 待决 "<标题>"` 提出。正文写清：背景、可选方案、推荐方案、影响哪些文档和哪些线、等裁决期间你先做什么。
2. 不要停下来等。先做不依赖裁决的部分；必须依赖的，先按推荐方案写在本线代码里，并在 `状态.md` 标明「依赖 D-xxx」。
3. 用户裁决后，由提出方取 `docs` 锁（涉及配置再取 `config` 锁），一次改完所有受影响的文档、`总揽.md` §7 和 `config.yaml`（`CLAUDE.md` 的规则）。文档里的代码块要按 `CLAUDE.md`「Verifying doc code」重新执行。改完给全体发「通知」。

两个例外可以先做后报：

**在本线自己负责的 `config.yaml` 小节里新增 key**（不改、不删已有 key）：

1. 取 `config` 锁，同一次提交里改好 `config.yaml`、`config.py` 的模型和 `tests/test_config.py`。
2. 取 `docs` 锁，在本线文档的参数说明处加一句含义，并写明 key 的位置。
3. 发一条 `待决` 请用户追认，再给全体发一条 `通知`。

**新增第三方依赖**：

1. 先确认包名拼写和维护状态，版本精确钉死（`==`）。
2. 持 `deps` 锁执行：`claude-codex/board.sh run deps L1 -- bash -c 'uv add "pkg==x.y.z" && make test'`。开发用的依赖加 `--dev`。
3. 立刻在 `git` 锁里单独提交 `pyproject.toml` 和 `uv.lock`。
4. 给全体发「通知」，写明包名、版本和用途。大于 500 MB 的依赖（例如 torch）同时发一条 `待决` 请用户追认。

## 7. Git

同一个暂存区被四个窗口共用。下面的规则是为了不毁掉别人还没提交的改动。

- 都在 `main` 上提交（沿用 P0 的做法）。每完成一个阶段、测试通过后提交一次。
- 只提交自己的文件，而且要在 `git` 锁里一次完成：

  ```
  claude-codex/board.sh run git L2 -- bash -c 'git add -- <文件...> && git commit -m "<提交说明>" -- <文件...>'
  ```

  `git commit -- <路径>` 只提交列出的路径，别的线暂存的内容不会被带进来。
- 提交说明的写法照旧：`L2 retrieval：build_expr 白名单与混合检索`，空一行后写要点。
- **禁止**：`git add .` / `-A` / `-u`、`git commit -a`、`git stash`、`git reset`、`git checkout -- <路径>`、`git restore`、`git clean`、`git rebase`、`git commit --amend`、切换分支、`git push`。这些命令都会碰到别的线的工作区。
- 不提交 `claude-codex/` 下的运行文件、`.planning/`、`真实内容按量读取/`、`data/`、`.env`。

## 8. 测试

- 开发时只跑自己的：`uv run pytest tests/<自己的目录>`。
- 提交前跑 `make test`。如果失败只出在别的线还没提交的文件里：给那条线发消息；只要 `uv run pytest tests/test_*.py tests/<自己的目录>` 全绿，就可以提交，并在提交说明里写明。
- 依赖外部服务的测试（Milvus、GPU、真实 LLM），服务不可用时用 `pytest.mark.skipif` 跳过并写明原因，不能让 `make test` 变红。不要新增自定义 marker，那要改 `pyproject.toml`。
- 「完成」必须有真实的测试输出作为依据。跳过的测试不算通过。要在 `状态.md` 里写清哪些项因为缺什么而没有验证。

## 9. 共享资源

| 资源 | 规则 |
|------|------|
| 端口 | 测试里起桩用 `tests/stubs/serve.py` 的 `serve()`，它会自动选空闲端口。要单独起服务时，L1 用 181xx，L2 用 182xx，L3 用 183xx，L4 用 184xx。8001（`services.embed.port`）留给 L1 的真实编码服务。 |
| Milvus collection | 每条线只建、只删自己前缀的 collection：`l1_*`、`l2_*`、`l3_*`、`l4_*`。别名 `rag_chunks` 只由 L1 在 M1b 完成后创建或切换，并通知全体。不要 drop 或 release 别人的 collection。不要跑 `scripts/milvus_bench.py`，它会 release 所有 collection。容器的 `up` / `down` / `restart` 只由用户操作。 |
| GPU | 只给 L1 用（Visualized-BGE 和 bge-reranker-v2-m3）。其他线一律用桩。 |
| 真实 LLM 接入点 | key 在 `.env`，用 `uv run --env-file .env …` 加载，不要把值打印出来。联调时每条线累计 20 次调用以内可以直接做，内容只用合成文本。批量调用（出评测集、完整档评测、RAGAS 验证）要等 D-002。 |
| MinerU 云端 | 上传文件要等 D-001。没批之前，用按 `数据清洗.md` §7 手写的 content_list 做测试。 |
| Langfuse 云端 | 只能在 `mask` 开着的情况下上报，冒烟测试最多发几条 trace。 |
| 模型下载 | L1 可以从 HuggingFace 下载 bge-m3、Visualized-BGE 权重和 bge-reranker-v2-m3，放进 `models/`（已 gitignore）。下载前在 `状态.md` 里写明。 |
| 真实语料 `真实内容按量读取/` | 只读，不提交，不复制进 `tests/`。按需抽样读，不要把整个目录读进上下文。其中的 md 文件走本地解析，不上传。 |

## 10. 跨线接口

接口以文档里的签名为准。消费方不用等，先按文档签名写代码，测试里用桩或假实现顶着。提供方做好后发一条「交付」消息，写明模块、签名和提交号。实现时发现签名必须和文档不一样，按 B 类走。

| 提供方 | 接口 | 出处 | 消费方 | 优先级 |
|--------|------|------|--------|--------|
| L3 | `llm/structured.py` 的 `structured_call` | 大模型生成 §6.2 | L2 预处理、改写；L4 出题 | L3 第一批 |
| L3 | `prompt/env.py`：Jinja 环境、`escape_tags`、`attr`、`RESERVED` | Prompt 组装 §5 | L2 渲染 `preprocess_*.j2` / `rewrite_*.j2` | L3 第一批 |
| L3 | `graph/deps.py` 的 `Deps`，节点依赖的注入方式 | D-003 | L2、L3 的所有节点 | L3 第一批 |
| L1 | `ingest/registry_db.py`：`documents` 表和 `doc_titles` FTS5 | 数据清洗 §12，用户问题预处理 §8 | L2 范围识别；L3 原文下载接口 | L1 第一批 |
| L1 | `ingest/chunking/tokenizer.py`：bge-m3 计数 | 文档切分 §15 | L3 Prompt 预算估算（以 `count` 参数注入） | L1 第一批 |
| L4 | `monitor/metrics.py`、`monitor/langfuse.py` | 评估与监控 §9、§10 | L2、L3 的节点埋点；L3 的 `api/app.py` | L4 第一批 |
| L4 | `monitor/feedback.py` 的反馈路由 | 评估与监控 §11 | L3 的 `api/app.py` 挂载 | L4 |
| L1 | 真实编码 / 重排服务（8001，协议和桩一样） | 向量化建库 §4.2 | 联调时所有线 | L1 后期 |
| L1 | 别名 `rag_chunks` 指向真实数据 | 向量化建库 §8 | L2 联调；L4 出题和评测 | M1b 之后 |
| L2 | `retrieve` / `rerank` / `build_context` / `preprocess` / `clarify` / `rewrite` 节点和路由函数 | 检索与重排 §12，用户问题预处理 §13 | L3 的 `graph/build.py`；L4 快速档 | L2 |
| L3 | `graph/build.py` 的 `build_graph`、`api/app.py` | 总揽 §4.2 | L4 完整档 | L3 后期 |

## 11. 完成

- 每条线的完成标准写在自己的任务书里。
- 完成后：在 `状态.md` 标 `done`，给全体发一条「交付」消息，写明交付清单、提交号和未验证项，然后停下。
- 不要为了「帮忙」去改别的线的文件。要帮忙，先发消息问对方，对方在消息里明确同意后才能动。
