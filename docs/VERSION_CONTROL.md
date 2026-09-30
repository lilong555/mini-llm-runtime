# Version Control

## Repository Policy

- Repository: `lilong555/mini-llm-runtime`.
- 可见性：保持公开，GitHub `visibility` 为 `PUBLIC`；仅在用户明确要求时更改。
- 公开交付范围：项目源码、测试、文档和符合 [产物政策](ARTIFACT_POLICY.md) 的发布证据。
  提交与 Release 发布前均须检查敏感内容；公开仓库不意味着可以上传本地秘密或受限产物。
- Default branch: `main`.
- Dependency checkout: fetched by `scripts/Fetch-Dependencies.ps1`, not
  vendored or silently updated.
- Model weights: fetched and verified through `models/manifest.json`, never
  committed. The derived numerical reference has its own manifest.
- Auxiliary Python experiments, credentials, local service state and compiled
  artifacts are excluded from source control.
- Trace `.jsonl` files preserve exact bytes through `.gitattributes`; their
  raw digests must continue to match recorded benchmark reports.
- CI actions are pinned to reviewed release commit IDs. Hosted runner OS
  labels are explicit; toolchain image updates still require fresh CI.

## Working Changes

Use one topic branch per coherent task, for example `feat/cpu-prefill-tiling`,
`fix/prefix-reclamation`, or `test/long-context`.

```powershell
git switch main
git pull --ff-only
git switch -c feat/cpu-prefill-tiling

.\scripts\Build-LLMServe.ps1
ctest --test-dir build/cpu --output-on-failure
.\scripts\Test-CtestEvidence.ps1
git diff --check
git status --short
git diff
git add <explicit-paths>
git diff --cached --stat
git diff --cached
git commit -m "perf(runtime): tile quantized prefill matrices"
git push -u origin feat/cpu-prefill-tiling
```

Run `ctest` from a Visual Studio developer shell, or use the CTest executable
beside the CMake installation selected by the build script. For model or
serving changes, also run the relevant commands in `docs/VALIDATION.md`.

Commit messages use `type(scope): summary`, with types such as `feat`, `fix`,
`perf`, `test`, `build`, and `docs`. Keep a commit limited to a coherent
behavioral change and its tests or evidence. The issue log is a current
problem register, not a replacement for Git history.

## Version Tags

The CMake project version follows `MAJOR.MINOR.PATCH`.
当前候选版本为 `0.2.0`，尚未发布；2026-09-30 查询时 `v0.2.0` 未占用。
发布前须再次检查标签与 Release，并以最终 clean candidate 的自身 CI 和
适用本机 smoke 为门禁。[发布草稿](RELEASE_DRAFT.md) 不代表门禁通过。
仅在用户明确授权后创建并推送 annotated tag、公开发布 Release；
不得移动既有研究标签。

A release candidate must have:

- A clean worktree and no staged credentials or large binary artifacts.
- Passing required CI jobs for the exact commit, not an older revision.
- Locally verified model and HTTP results for behavior that CI does not run.
- Current limitations and open problems recorded in the documentation.
- Explicit separation between CPU, upstream CUDA, and custom kernel claims.

上传前检查暂存文件的路径和大小，不得提交凭据或其他本地秘密。
仓库创建或设置变更后，核验 GitHub 的 `visibility` 字段为 `PUBLIC`。

## 历史分支清单

2026-09-30 已获取的 origin refs 相对 `origin/main@9a571a5`：

| 分支 | 落后 main | 独有提交 |
| --- | ---: | ---: |
| `build/wsl-native` | 42 | 0 |
| `feat/cuda-paged-kv` | 2 | 0 |
| `feat/own-cuda-serving` | 24 | 0 |
| `feat/own-cuda-vertical-slice` | 30 | 0 |
| `fix/http-shutdown-drain` | 23 | 0 |
| `fix/v-value-simd` | 45 | 0 |
| `fix/windows-ci` | 28 | 0 |
| `perf/cuda-f16-matrix-path` | 17 | 0 |
| `release/own-cuda-serving-public` | 16 | 0 |

这些已获取的分支头均可从 main 到达；清单不是删除授权。
本轮保留全部 refs。实际删除前仍须重新核对远端、活跃 PR 和本地独有工作，
不得把本轮尚未提交的收尾修改当作可丢弃状态。
