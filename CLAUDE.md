# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

This repo is a RAG Q&A system (LangGraph + Milvus DISKANN). P0 and the four parallel lines L1–L4 (`docs/开发计划.md` §3–§4) are done, and the four-window protocol ended on 2026-10-03; work is now a single session on `main`. Milestone status is in `开发计划.md` §6: M1b is done as a local pipeline (MinerU cloud parse quality confirmed, D-018 B closed), M4 passed on 2026-10-19 (471-question set, 40-question stratified baseline; the full 471-question run is deferred to tuning), and M5 is a single-host plan in `docs/服务化部署.md` (systemd user units in `deploy/systemd/` for API + embed, compose for Milvus; monitoring stack, auth, backups and an ingest lock deferred). It holds:

- 13 Chinese design docs in `docs/`. `docs/总揽.md` is the entry point. It has the architecture, the LangGraph state and graph, the tech stack, the directory tree (§6) and the decision log (§7).
- `config.yaml`: every tunable parameter for the whole system. `src/ragagent/config.py` models it and validates it at startup.
- `prompts/v1/*.j2`: the real Jinja2 prompt templates.
- `src/ragagent/{ingest,index,retrieval,preprocess,context,prompt,llm,postprocess,graph,api,monitor}`: the implemented packages. Each doc's "目录与模块" section says what goes where. `api/static/` is the same-origin demo page (`总揽.md` §4.3).
- `eval/`: question generation, the quick/full eval CLI, RAGAS scoring and calibration (`eval/README.md`). Reports and datasets built from the real corpus stay gitignored.
- `tests/`: unit and integration tests, contract tests (every doc model must match its `models.py` definition via AST), fixtures in `tests/fixtures/`, and three stub services in `tests/stubs/` (embed + rerank, OpenAI-compatible LLM).
- `docker-compose.milvus.yml` + `milvus.yaml`: Milvus 2.5.18 standalone. MinIO credentials come from `.env`. The production collection name `rag_chunks` is an alias for the real index (D-019).
- `claude-codex/`: the archived board, status and decision files of the parallel lines. Read-only history now.

Talk to the user in Chinese. Docs are also written in Chinese.

Commands (Python 3.12 via uv; don't invent others):

- Tests: `make test` (= `uv run pytest`).
- Milvus: `docker compose -f docker-compose.milvus.yml up -d` / `down`. etcd, MinIO and Milvus metadata are in `volumes/` (gitignored). Milvus's local cache (`localStorage.path`: mmap and DiskANN files) is in the named volume `milvus_local`, because on Docker Desktop bind mounts go through virtiofs and make mmap loads about 80x slower (向量化建库 §6.4).
- Milvus smoke: `uv run --env-file .env python scripts/milvus_smoke.py`. It uses its own `rag_chunks_smoke` collection.
- Milvus measurements (向量化建库 §6.4): `uv run --env-file .env python scripts/milvus_bench.py {mmap,recall}`. They take minutes and release other loaded collections.
- Embed + rerank service (GPU, port `services.embed.port`): `uv run --env-file .env python -m ragagent.index.embed_service`.
- API (needs Milvus and the embed service): `uv run --env-file .env uvicorn ragagent.api.app:create_app --factory --host 127.0.0.1 --port 8000`. The demo page is `http://127.0.0.1:8000/`. There is no auth, so bind only to localhost or a trusted intranet.
- Eval: `uv run python -m eval.run --tier {quick,full} --fixture` offline; real runs need `--env-file .env`, `--chunks` and an explicit `--max-calls` (`eval/README.md`).
- Stubs standalone: `uv run python -m tests.stubs.embed_server` (also serves rerank), `tests.stubs.rerank_server` (8002), `tests.stubs.llm_server` (8003). All take `--port`; the embed stub defaults to `services.embed.port`.

## Parallel lines (ended)

L1–L4 ran as four Claude Code windows from 2026-10-01 to 2026-10-03 under `claude-codex/README.md`. The protocol is over: don't post to the board or take its locks. The general git rules still hold: commit only specific paths, and never `git add .`/`-A`, `stash`, `reset`, `checkout --`, `restore`, `clean` or `--amend`.

## How design work is done here

- **Discuss first, then write.** For any design change or new doc, present the plan and defaults, ask the open decisions, and write only after the user confirms. Don't ask whether to discuss first; go straight into the discussion.
- **When the user's choice affects an earlier doc**, say so and confirm before editing. Then update every affected doc, the decision table in `总揽.md` §7 and `config.yaml` in the same pass.
- **Parameters go only in `config.yaml`.** Docs explain what a parameter means and point to its config key. Don't paste YAML snippets with values into docs.
  - Secrets appear only as `*_env` variable names. Every such name must also be listed in `.env.example`. The real `.env` is gitignored.
  - The exception is Milvus server settings (DISKANN build params, mmap). They live in Milvus's own `milvus.yaml`, documented in `向量化建库.md` §6.

## Verifying doc code

Code blocks in the docs are drafts, but they have been executed. Keep it that way when you edit them.

1. Extract the fenced `python` blocks from the md file with a regex and `exec` them. Stub any external SDKs (openai, pymilvus, langchain).
2. Run the extraction script from `/tmp` with a pinned temporary env. System Python is 3.14, but the project targets 3.12. For example:
   `uv run --quiet --python 3.12 --with pyyaml==6.0.2 --with jinja2==3.1.6 python /tmp/check.py`
3. Check YAML with a SafeLoader subclass that rejects duplicate keys.
4. Check every doc for unbalanced code fences:
   `for f in docs/*.md; do n=$(grep -c '^[ \t]*```' "$f"); [ $((n % 2)) -ne 0 ] && echo "odd: $f"; done`
   Inside a code block, never write three literal backticks. Use `` `{3,} `` or `"`" * 3`. This broke `后处理.md` once.
5. Delete temp scripts afterwards.

Also note that `Prompt 组装.md` has a space in its filename. Quote it in shell commands, and link to it as `Prompt%20组装.md`.

## Cross-file invariants

Each of these contracts spans several docs. Change one side and you must change the others.

- **Chunk schema.** Defined in `文档切分.md` §4 and mirrored as Milvus fields in `向量化建库.md` §5. It is used by `检索与重排.md` §6, `上下文构造.md` and `评估与监控.md` §3.4.
  - `chunk_id` is `{doc_id}:v{version}:{seq:05d}`, so it is deterministic.
  - `doc_id` is 16 hex characters and must be regex-checked before it goes into any Milvus filter expression.
  - Build and parse it only with `make_chunk_id` / `parse_chunk_id` (`文档切分.md` §4).
  - Other fields: `tokens`, `section_idx`, `section_tokens`, `prev_id`/`next_id` (text and code only), `image_ids`, `related_ids`, `searchable`, `content_hash`.
  - `content_hash` is the sha256 of the normalized `text` (`文档切分.md` §11). It is stored in Milvus and used for post-recall dedup (`检索与重排.md` §4.4). Duplicates within a doc and boilerplate get `searchable=false`. There is no `skip_index`.
  - `ChunkFields` (`向量化建库.md` §5.1) must match the fields in `create_collection` one to one. `Hit` and `Expansion` (`检索与重排.md` §4.5) derive from it.
- **Data models.** Every cross-module structure is a pydantic v2 `BaseModel`, and the doc definition matches the package's `models.py` word for word. Construct them with keyword arguments only. `RAGState` stays a TypedDict and holds `model_dump()` dicts.
  - `meta.overlap_chars` is the overlap between split parts. Context merge and eval alignment both strip it.
- **`embed_text`** is the heading path plus the text. It is the input for both the dense vector and BM25. `text` has no prefix and is what the LLM sees.
- **Filters.** User-influenced values reach Milvus expressions only through whitelist or regex checks.
  - `build_expr` in `检索与重排.md` §4.2 does this.
  - Soft filters come from `scope` in the question and are resolved to doc_ids via the registry DB, never taken from LLM text.
  - Hard filters come from API params.
- **Prompt tags.** The `RESERVED` tag list in `escape_tags` (`Prompt 组装.md` §5) must include every tag used in `prompts/v1/*.j2`.
  - Escape only those tags. Never HTML-escape, because it breaks code in answers.
  - Jinja runs with autoescape off, `StrictUndefined` on, and the `esc` and `attr` filters.
- **Supplement heading.** `prompt.supplement_heading` is used both by `system_rag.j2` and by the answer parsing in `后处理.md` §3.2.
- **Citations.**
  - `[n]` refers to a `ContextBlock.cite` (`上下文构造.md` §8).
  - Postprocess never edits the streamed text. It only emits a `final` event whose offsets are in UTF-16 units.
  - It also saves a `plain` copy with citations stripped. Prompt assembly uses `plain` for history.
- **Graph flow.** Defined in `总揽.md` §4.2.
  - Every answering path ends at `postprocess`, and `clarify` goes through it too.
  - Grade is a conditional edge after `rerank`.
  - `rewrite` first drops the soft filters before calling the rewrite LLM.
- **LLM access.** Everything uses OpenAI-compatible endpoints, defined in `llm.endpoints`. Each scene (generate, chat, preprocess, rewrite, eval) in `llm.scenes` names one primary and one fallback endpoint.
  - Retries happen only before the first visible token, and a first-token timeout switches endpoints immediately.
  - Preprocess and rewrite never retry. They fall back once, then degrade by rules.
- **Privacy.** Langfuse is the cloud service and receives metadata only. `_mask` in `评估与监控.md` §9.1 keeps a whitelist of fields and masks every other string. The eval set is LLM-generated, and its gold answers are stored as `doc_id` plus `quote`, never as chunk_id.

## Fixed constraints

- GPU: RTX 4060 with 8 GB. It runs Visualized-BGE (bge-m3 base, 1024 dimensions) and bge-reranker-v2-m3 locally.
- Scale: designed for 10M chunks. Milvus resident memory must stay at or below 10 GB, which is why PQ and cache ratios are reduced and mmap is on.
- No department or classification ACL. Excel is not supported. MinerU (cloud API) parses PDF, Word, PPT and images.

## Progress tracking

Track multi-step work with the `planning-with-files-zh` skill. The session keeps `task_plan.md` / `findings.md` / `progress.md` in the repo root (gitignored). `.planning/L{n}/` holds the archived plans and evidence of the parallel lines.
