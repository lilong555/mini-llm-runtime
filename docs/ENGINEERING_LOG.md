# Engineering Problem Register

This register records reproducible problems, their current disposition, and
the evidence for each solution. `Resolved` means the stated scope has been
verified; `Mitigated` means an operational workaround exists; `Open` requires
further work. This is not a release changelog.

## Index

| ID | Status | Area | Problem |
| --- | --- | --- | --- |
| ENG-001 | Resolved | Build | Localized MSVC include output breaks Ninja rules |
| ENG-002 | Resolved | Build | C++20 leaks into a C++17 dependency |
| ENG-003 | Resolved | GGUF | Optional RoPE metadata treated as mandatory |
| ENG-004 | Resolved | Validation | Quantized reference includes activation quantization |
| ENG-005 | Mitigated | Download | Native HTTPS model download fails |
| ENG-006 | Resolved | Operations | Timestamp type mismatch blocks graceful shutdown |
| ENG-007 | Resolved | Lifecycle | Prefix-copy failure can publish duplicate terminal events |
| ENG-008 | Open | Performance | Mixed scheduling is worse on the measured CPU workload |
| ENG-009 | Resolved | Validation | Manually specified reference filename does not exist |
| ENG-010 | Open | Dependency | Pinned upstream emits MSVC warnings |
| ENG-011 | Open | Model | Tokenizer normalizes a control-looking vocabulary entry |
| ENG-012 | Resolved | Evidence | CTest truncates successful-suite output |
| ENG-013 | Resolved | Versioning | Text normalization changes benchmark trace digests |
| ENG-014 | Resolved | Evidence | PowerShell audit continues after an XML access error |
| ENG-015 | Mitigated | CI | Deprecated action runtime and moving OS labels |

## ENG-001: Localized MSVC Include Output

- Status: Resolved.
- Impact: Ninja cannot reliably parse dependency output from the localized
  MSVC compiler; generated build rules can be malformed.
- Reproduction: configure a Ninja build with MSVC whose `/showIncludes`
  prefix uses the Windows ANSI code page rather than UTF-8.
- Cause: dependency-prefix detection decoded compiler output incorrectly.
- Solution: `cmake/MSVCIncludes.cmake` probes a real include, decodes it with
  `ENCODING ANSI`, and sets `CMAKE_CL_SHOWINCLUDES_PREFIX`. Do not modify
  generated Ninja files.
- Verification: native MSVC 19.44 CPU Release configuration and all 112 build
  steps completed; CTest passed both registered suites. The CUDA build also
  completed on the same toolchain. Non-Windows builds bypass the probe.

## ENG-002: Dependency Language Standard

- Status: Resolved.
- Impact: the upstream tokenizer fails to compile when UTF-8 literals acquire
  C++20 `char8_t` semantics.
- Reproduction: add the pinned llama.cpp subtree under a parent project with
  `CMAKE_CXX_STANDARD=20`.
- Cause: the parent language setting is inherited by dependency targets.
- Solution: configure llama.cpp under C++17 and restore C++20 for project
  targets; compile the upstream HTTP vendor target as C++17 as well.
- Verification: CPU and CUDA product builds complete without modifying
  upstream source. `scripts/Fetch-Dependencies.ps1` verifies the clean pinned
  checkout.

## ENG-003: Optional Qwen3 RoPE Metadata

- Status: Resolved.
- Impact: a valid official Qwen3-0.6B GGUF was rejected during model loading.
- Reproduction: load the artifact pinned by `models/manifest.json`; it omits
  `qwen3.rope.dimension_count`.
- Cause: an optional metadata field was required unconditionally.
- Solution: use the explicit attention head dimension and validate the RoPE
  dimension only when the optional key exists. Still reject incompatible
  shapes and unsupported partial rotary dimensions.
- Verification: real-model CPU/CUDA-reference logits and greedy output checks
  pass. Chunk boundaries 1, 7 and 16 and a 17-token prefix branch preserve
  MiniLLM logits.

## ENG-004: Matched-Weight Numerical Reference

- Status: Resolved.
- Impact: a direct comparison to llama.cpp using Q8_0 weights exceeds the
  strict logits thresholds even when the weights and model architecture match.
- Reproduction: run `llmserve-model-tests` with the Q8_0 file as both the
  implementation and reference model.
- Cause: upstream quantized matrix kernels additionally quantize activations;
  MiniLLM multiplies Q8_0 weights by F32 activations. The two arithmetic paths
  are not the same numerical oracle.
- Solution: dequantize the exact Q8_0 weights to F32 with the pinned upstream
  converter, verify the derived file hash, and use this matched-weight
  reference. Do not relax acceptance thresholds to hide the discrepancy.
- Verification: ten checks pass against both CPU and CUDA F32 references.
  Teacher-forced CPU-reference RMSE is 0.000963-0.002472. See
  `benchmarks/results/validation/model-f32-cpu-reference.json` and
  `model-f32-cuda-reference.json`. Diagnostic Q8 reports are retained under
  `benchmarks/results/diagnostics/`.
- Boundary: this does not establish Q8-backend equivalence for every prompt,
  restore original pre-quantization weights, or validate all model variants.

## ENG-005: Native HTTPS Model Download

- Status: Mitigated.
- Impact: the native Windows HTTPS path failed to download the model from
  Hugging Face in this environment.
- Cause: a native TLS/network-path failure was observed; its machine-level
  cause has not been isolated.
- Workaround: use `scripts/Download-Model.ps1 -UseWsl` with WSL curl, then
  verify size and SHA-256 against the pinned manifest before renaming the
  partial artifact. Do not disable TLS verification.
- Verification: the official model was downloaded through this path and
  passes the manifest checks.
- Next action: reproduce and diagnose the native TLS path independently.
  WSL is an optional download route, not a runtime dependency.

## ENG-006: Shutdown Process Identity

- Status: Resolved.
- Impact: a running server could not be stopped using its saved process
  identity even though the PID and executable were correct.
- Reproduction: load the service JSON with a PowerShell version that
  deserializes an ISO timestamp as `DateTime`, then compare its string form
  with a freshly formatted timestamp.
- Cause: JSON timestamp conversion changed representation without changing
  the instant, so a string comparison rejected the correct process.
- Solution: compare UTC ticks and the executable path before creating the
  shutdown marker. Verify that the marker is inside the project `.run`
  directory. Preserve identity checking so a reused PID is not targeted.
- Verification: `scripts/Stop-LLMServe.ps1` gracefully stopped each of the six
  CPU policy trials; those trials use separate server processes.

## ENG-007: Prefix-Copy Failure Finalization

- Status: Resolved.
- Impact: a backend exception during prefix reuse could reach more than one
  terminal-handling path, duplicating completion and resource accounting.
- Reproduction: inject a failure from `ModelRunner::copy_sequence()` after a
  reusable prefix has entered the cache.
- Cause: admission failure and fail-stop cleanup can both finalize a handle.
- Solution: make terminal publication idempotent and ensure cleanup releases
  reservations and cache references.
- Verification: the regression
  `engine_prefix_copy_failure_has_one_terminal_and_no_reservation_leak`
  asserts one terminal event, one failed request, no second event, and zero
  used capacity credits after shutdown. It passes in the 28-case CPU unit
  suite; see `benchmarks/results/validation/windows-cpu-ctest.xml`.

## ENG-008: CPU Mixed-Scheduling Regression

- Status: Open.
- Impact: mixed prefill/decode does not improve throughput or latency for the
  measured CPU pressure workload.
- Reproduction: replay `benchmarks/traces/cpu-mixed-s0.jsonl` with
  `scripts/Benchmark-Policies.ps1 -Backend mini`; use three trials per policy,
  24 requests per trial, 4 requests/s, 128/16 prompt tokens, and 16 output
  tokens. Restart and warm the server identically for each trial.
- Evidence: `benchmarks/results/mini-scheduling/summary.json`; all 144 requests
  succeeded and no output-token sequence mismatch was observed.

| Median of three trials | Mixed | Prefill-first |
| --- | ---: | ---: |
| Output tokens/s | 18.40 | 18.59 |
| P95 TTFT, ms | 13105.77 | 12290.59 |
| P95 mean TPOT, ms | 398.89 | 352.85 |
| Goodput, requests/s | 0 | 0 |

- Cause: not yet established. CPU prefill cost, matrix weight reuse and
  token-budget composition are hypotheses, not profiling conclusions.
- Next action: collect per-stage CPU profiles with the fixed trace; isolate
  matrix multiplication, attention, scheduling and waiting. Compare cold and
  warm prefix workloads and a lower arrival rate. Optimize one identified
  bottleneck, retain the unfavorable baseline, and repeat numerical and
  end-to-end checks before claiming a speedup.
- Acceptance: reproducible gains on a declared workload without output
  changes or unacceptable fairness/tail-latency regressions. A SIMD dot
  microbenchmark alone cannot close this issue.

## ENG-009: Reference Artifact Path

- Status: Resolved.
- Impact: a model test invocation failed before running any numeric checks.
- Reproduction: request `Qwen3-0.6B-from-Q8_0-F32.gguf`, which is not the
  artifact filename stored in the reference manifest.
- Cause: a manually transcribed path diverged from
  `models/reference-manifest.json`.
- Solution: use `scripts/Validate-Model.ps1`; it resolves both model paths
  from manifests and verifies file size and SHA-256 before execution. Select
  the report destination with `-Output`, not by rebuilding model filenames.
- Verification: the missing-file invocation exits nonzero and is retained
  under `benchmarks/results/diagnostics/reference-path-mismatch.json`.
  Manifest-based validation verifies both artifacts and passes all ten model
  checks in `benchmarks/results/validation/model-f32-cpu-only-reference.json`.

## ENG-010: Upstream Compiler Warnings

- Status: Open.
- Impact: the pinned llama.cpp dependency emits MSVC C4297, C4244 and C4834
  warnings during the CPU Release build. The project cannot claim a
  warning-free dependency build.
- Reproduction: `scripts/Build-LLMServe.ps1` with MSVC 19.44.
- Evidence: warnings originate in upstream load-mode conversion, graph/model
  integer conversions, quantization argument forwarding and sampler code;
  the build succeeds. Project translation units emitted no warnings in this
  build.
- Cause: upstream exception specifications, narrowing conversions and ignored
  `nodiscard` results. No failure in the scoped model tests is attributed to
  these warnings.
- Next action: inspect relevant upstream fixes at a chosen future revision,
  then re-pin and rerun build, model, HTTP and benchmark checks. Do not edit
  the dependency checkout or suppress warnings globally to conceal them.

## ENG-011: Model Vocabulary Warning

- Status: Open.
- Impact: the pinned tokenizer warns that token 128247, `</s>`, looks like a
  control token but is not declared as one in the model vocabulary.
- Reproduction: load the official pinned GGUF with the CLI or model tests.
- Cause: an upstream heuristic disagrees with the artifact's token metadata;
  whether the artifact or the heuristic should change is not established.
- Current behavior: the pinned tokenizer normalizes the token type during
  loading. MiniLLM and its numerical reference share this tokenizer. The
  focused numeric, generation and HTTP checks pass, but they do not establish
  complete special-token compatibility.
- Next action: add explicit vocabulary/special-token regression cases and
  check upstream model metadata reports. Do not silently edit the downloaded
  artifact or change its manifest hash.

## ENG-012: CTest Report Truncation

- Status: Resolved.
- Impact: the default successful-test stdout limit omitted the final cases
  and case count from the JUnit artifact, reducing the usefulness of the
  saved evidence despite a passing suite result.
- Reproduction: export the 28-case unit runner through CTest with its default
  1024-byte passed-output limit.
- Cause: CTest treats the whole unit executable as one test and truncates
  its captured output.
- Solution: set `--test-output-size-passed 65536` when saving JUnit evidence;
  CI uploads the report even on test failure.
- Verification: `benchmarks/results/validation/windows-cpu-ctest.xml` includes
  all 28 unit-case results and all four GGUF-case results.

## ENG-013: Byte-Identical Benchmark Traces

- Status: Resolved.
- Impact: automatic CRLF-to-LF normalization changes the replay file's raw
  digest even though its parsed requests remain identical. A cloned trace
  would no longer match the digest in the archived benchmark reports.
- Reproduction: compare `git hash-object --no-filters` for the measured
  `benchmarks/traces/cpu-mixed-s0.jsonl` with its staged or committed blob.
- Cause: the generic text normalization rule also applied to `.jsonl`
  traces; `llmserve-bench` hashes raw bytes, including newline characters.
- Solution: declare `*.jsonl -text` in `.gitattributes` and stage the original
  trace bytes without normalization. Apply `whitespace=cr-at-eol` to traces
  so `git diff --check` accepts their intentional CRLF line endings. Keep
  ordinary source files normalized.
- Verification: the raw file and Git blob both have object ID
  `585b02f75cb8cc9242f57aca141099c57e7f5931`. Replay request data and the
  original benchmark reports are unchanged.

## ENG-014: Fail-Closed Evidence Inspection

- Status: Resolved.
- Impact: an ad hoc archive-inspection command accessed `XmlDocument.InnerText`
  as though it were an element body. PowerShell reported a null-method error
  but continued to subsequent Git commands with its default error policy.
  The remote CI jobs themselves had passed; the extra local audit was
  incomplete.
- Cause: incorrect XML node access combined with non-terminating script
  errors. PowerShell's XML adapter can also shadow native property names
  with XML attributes.
- Solution: `scripts/Test-CtestEvidence.ps1` uses a parsed `XmlDocument`,
  explicit element selection and native XML getters, strict mode, and
  `$ErrorActionPreference = 'Stop'`. It checks suite totals, failure/skipped
  states, full passed-case counts, and missing or truncated output.
- Verification: all six archived local/remote reports pass, representing
  nine suite executions and 180 case executions. Failed and truncated
  fixture reports are both rejected. These are repeated case executions,
  not 180 distinct tests.

## ENG-015: GitHub Actions Runtime Compatibility

- Status: Mitigated; the pinned workflow is validated by its exact-commit
  GitHub Actions run before a version tag is published.
- Impact: the runner warned that `actions/checkout@v4` and
  `actions/upload-artifact@v4` target deprecated Node 20 and are being forced
  onto Node 24. It also announced a future migration of `ubuntu-latest`.
- Evidence: annotations on run `35354736795`. The C++ unit and sanitizer
  checks passed, but the CI environment was not fully pinned.
- Solution: use the official v7.0.1 releases, whose `action.yml` files
  explicitly declare `node24`. Pin checkout to
  `3d3c42e5aac5ba805825da76410c181273ba90b1` and upload-artifact to
  `043fb46d1a93c77aae656e7c1c64a875d1fc6a0a`. Use `ubuntu-24.04` and
  `windows-2025` labels, and reject missing test artifacts.
- Verification: release tags, commit IDs and runtime declarations were
  checked against the official `actions` repositories. The workflow's
  exact-commit run remains the build/test acceptance gate.
- Boundary: named hosted OS versions still receive image updates. Fully
  bit-reproducible toolchain images are not claimed.
