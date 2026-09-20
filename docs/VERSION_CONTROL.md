# Version Control

## Repository Policy

- Repository: `lilong555/mini-llm-runtime`.
- Visibility: public. Changes to repository visibility still require an
  explicit request from the owner.
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

The CMake project version follows `MAJOR.MINOR.PATCH`; the initial version is
`0.1.0`. Use annotated tags such as `v0.1.0` only on an inspected, tested
commit. Push tags explicitly. Never force-update a published tag.

A release candidate must have:

- A clean worktree and no staged credentials or large binary artifacts.
- Passing required CI jobs for the exact commit, not an older revision.
- Locally verified model and HTTP results for behavior that CI does not run.
- Current limitations and open problems recorded in the documentation.
- Explicit separation between CPU, upstream CUDA, and custom kernel claims.

Before any upload, inspect staged paths and sizes. Never commit credentials or
other local secrets. After repository creation or a settings change, confirm
the GitHub `visibility` field is `PUBLIC`.
