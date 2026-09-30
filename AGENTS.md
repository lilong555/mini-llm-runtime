# Project Rules

- Feature Freeze：只允许文档、清理、发布准备和有最小复现的必要修复。
  当前收尾合同为 `docs/finalization/FINALIZATION_SPEC.md`；
  不新增主要功能，不修改 Runtime 数学、调度或 CUDA 完成契约，不追加性能 trial。
  新版本公开发布及分支删除仍需用户明确授权。

- This is a C++20 Mini LLM Runtime and serving project. Keep the runtime,
  serving code, and upstream llama.cpp ownership boundaries explicit.
- Deliverables describe the current usable system. Do not add patch notes,
  editing narratives, or explanations of the editing process to the README.
- Record encountered engineering problems in `docs/ENGINEERING_LOG.md`.
  Write its titles, field labels, statuses, and explanations in Simplified
  Chinese. Preserve issue IDs, code identifiers, paths, commands, and original
  diagnostic messages.
  Each entry needs an ID, status, impact, reproduction or evidence, cause,
  solution or next action, and verification. Never mark an issue resolved
  without evidence. Keep credentials and tokens out of the log.
- 仓库保持公开，GitHub 可见性应为 `PUBLIC`。源码、文档和经检查的发布证据
  按公开交付范围处理；仅在用户明确要求时更改仓库可见性。
  公开发布不豁免凭据、模型权重、构建产物和本地运行状态的排除规则。
- Use scoped commits on topic branches after the initial baseline. Do not
  rewrite shared history or move published version tags.
- Do not commit model weights, build output, dependency checkouts, local
  service state, or the auxiliary Python experiments. Pin dependencies and
  model provenance through source-controlled metadata.
- Run CTest for core changes. Model, numeric, KV, and serving changes also
  need the relevant real-model or HTTP checks. Keep raw benchmark evidence
  and report unfavorable results as well as improvements.
- Do not equate SIMD microbenchmarks with end-to-end speedups, or llama.cpp
  GPU execution with a custom CUDA paged attention implementation.
