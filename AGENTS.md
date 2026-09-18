# Project Rules

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
- Keep this repository private unless the user explicitly requests otherwise.
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
