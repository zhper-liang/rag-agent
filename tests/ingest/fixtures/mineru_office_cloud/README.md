# MinerU 真实云端返回：合成 Office 离线回归夹具

2026-10-02 从已有 vlm / pipeline 批次恢复的真实 `content_list.json`，共 8 份：DOCX、PPTX 各 2 个自生成样本 × 2 种模型。仅包含自生成文字，不含真实用户语料。上传已获 D-001 授权；此导出没有新提交或上传。

- `manifest.json` 记录原合成文件 hash、返回原文 hash、模型和已核验页数。
- `*.content_list.json` 是返回原文，不包含 token、签名 URL、原始 Office 二进制或账户响应。
- `*.blocks.json` 固定完整 Block 快照，另外用显式文字/层级/表格/页码断言独立核验。
- 两种模型目前的 Office 返回相同，不代表 VLM 质量优于 pipeline。
- DOCX 的实际源文件 XML 表格只有一列（Model/BGE-M3/Reranker）；不能把生成脚本原打算导出的两列当成实际源文件、误报云端丢列。
- PPTX 返回没有 `text_level`，标题文字会是 paragraph，不虚构 heading。
- 样本中的 `Formula: ...` 只是普通文字，没有 equation 返回；此夹具不能证明数学公式识别通过。
- 完整清洗后，PPT 中两页重复的 Deployment / 中文正文命中了页眉页脚去重规则，被过滤；这是待人工复核的启发式风险，不把缓存/管线成功当质量签署。详见 `.planning/L1/office-cloud-quality-review.json`。
- 没有图片返回，图片真实落盘验收、PDF/PNG/JPEG 夹具及完整两模型质量对照仍待既有批次完成；本组仅验收 Office 返回。

测试中的小字节源文件与 ZIP 包装是显式离线替身，仅用于缓存及管线回归，不能冒充真实 Office / 云端调用。真实源文件恢复后的完整 `run_clean` 证据另存于 `.planning/L1/`，不提交运行数据。
