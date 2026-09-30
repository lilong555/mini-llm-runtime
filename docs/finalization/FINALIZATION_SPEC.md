# FINALIZATION_SPEC

## 0. 执行合同

Baseline：`main@9a571a5d7a6570c6dfeeac9062adefca1840534e`。

允许：cleanup、documentation、已证明必要的小集成修复、release preparation。
禁止：新模型、新精度产品路径、新 attention、Graph/async、多卡、通用 allocator/plugin/error framework、重新设计 Runtime/Scheduler/HTTP。

开始时记录当前 HEAD、分支、工作树状态及 delta。不得覆盖用户未提交或未推送工作；不得自动 stash、reset、clean、checkout 其它分支或 force-push。可以读取原工作树；需要构建时优先经确认的隔离目录。仅有访问授权不等于自动获准删除分支或公开发布。

本次审计没有进入用户 WSL。下列命令是实施规范，不是“已经在本机跑过”的记录。

## T01 — 当前事实一致性 / P0

**Files**：`README.md`、`THIRD_PARTY.md`、`docs/GPU_KV_STUDY.md`、`docs/CUDA_SERVING.md`、`docs/EXECUTION_STATUS.md`、`AGENTS.md`；其它文档仅在发现同类当前事实冲突时修改。

**Why**：当前页面否认已实现的 GPU Serving，或把已完成研究写成尚未采集。

**Change**：当前支持范围置顶；写清 main 已整合、contiguous 默认、paged 为 B、FP16 Serving 禁止。旧阶段正文冠以日期/采集身份，保留原结果和失败。AGENTS 增加短 Feature Freeze 规则，并把当前实施入口指向本次收尾规范而非旧 Feature roadmap。

**Tests**：逐处核对源码与当前 GitHub 元数据；搜“没有自有 GPU Serving”“尚未采集”“未合并 main”“活跃计划”，判断所在段是当前说明还是明确历史，不做无差别替换。

**DoD**：主入口没有相互矛盾的当前事实；历史原始记录未改写；不把新 main 冒充旧实验采集 SHA。

## T02 — Golden Path / P0

**Files**：`README.md`、`docs/WSL_DEVELOPMENT.md`、必要时 `scripts/dev.sh`。

**Why**：陌生人不能复制作者绝对路径；完整测试依赖与 CUDA 架构需提前知道。

**Change**：从 clone 开始；Linux 路径使用用户自行选择的工作区。提前列 Git、CMake≥3.24、C++20 compiler、Ninja、Python、PowerShell；CUDA 加 Toolkit≥12.8、适配驱动/GPU/架构。指出 CUDA_ARCHITECTURES 当前默认 89 是已验证机器设置，不代表通用 GPU。

只补已有入口的短诊断，不自动安装系统包，不新增 doctor 或脚本框架，不通过关闭 `LLMSERVE_REQUIRE_TEST_TOOLS` 规避完整验收。

CPU 快速入口：

```bash
git clone https://github.com/lilong555/mini-llm-runtime.git
cd mini-llm-runtime
bash scripts/dev.sh dependencies
bash scripts/dev.sh model
bash scripts/dev.sh build
bash scripts/dev.sh test
build/wsl-cpu/bin/mini-llm --model models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "The capital of France is" --tokens 8
bash scripts/dev.sh serve --port 8000
```

Own CUDA 入口，承接已准备的依赖/模型：

```bash
# 89 is the audited RTX 4070 Laptop architecture, not a universal default.
CUDA_ARCHITECTURES=89 bash scripts/dev.sh own-cuda build
bash scripts/dev.sh own-cuda test
bash scripts/dev.sh own-cuda generate --prompt "The capital of France is" --tokens 8
bash scripts/dev.sh own-cuda serve --port 8001
```

`serve` 前台运行；另一个终端访问：

```bash
curl --fail --silent http://127.0.0.1:8001/health
curl --fail -N http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' \
  --data '{"prompt":"The capital of France is","max_tokens":8,"temperature":0,"stream":true}'
curl --fail --silent http://127.0.0.1:8001/metrics
```

**Tests**：在新工作目录逐行执行，检查 CPU/CUDA backend、prefix capability、precision 与 layout；验证缺依赖时错误可理解。CPU 客户端使用 8000，不混用 CUDA 8001。

**DoD**：一个 CPU 入口、一个 Own CUDA 入口和共同服务访问说明；无人需要推断目录/终端/后端。不得承诺安装、模型下载和编译一定在 5～10 分钟内完成。

## T03 — 候选验证 / P0

**Files**：已有 CMake/测试/脚本只运行；新的本地结果放唯一 `.run/finalization-<identity>/`，不覆盖旧目录。

**Why**：本次只有远程源码和历史报告，没有用户 WSL 的实际验证。

**Change**：先只读盘点：

```bash
pwd
git rev-parse --show-toplevel
git status --short --branch
git rev-parse HEAD
git log -5 --oneline
git diff --stat
git diff --cached --stat
git ls-files | wc -l
git count-objects -vH
command -v cmake ninja python3 pwsh nvcc nvidia-smi
```

只读取与项目有关的环境，不转储全量环境变量、SSH 配置、凭据或其它目录。若存在本地修改，记录而不改动；明确审计 GitHub SHA 与 WSL HEAD 是否同一版本。

根据现有环境使用已有入口：

```bash
bash scripts/dev.sh build
bash scripts/dev.sh test
bash scripts/dev.sh own-cuda build
bash scripts/dev.sh own-cuda test
bash scripts/dev.sh own-cuda generate --prompt "The capital of France is" --tokens 8
```

短数值检查需要 derived F32 reference，不能只下载 Q8_0 就执行：

```bash
python3 scripts/models.py --reference --converter build/wsl-cpu/bin/mini-llm
OUT="$PWD/.run/finalization-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)-$$"
test ! -e "$OUT" || exit 1
mkdir -p "$OUT"
bash scripts/dev.sh own-cuda model-check "$OUT/model"
bash scripts/dev.sh own-cuda serving-check "$OUT/serving.json"
bash scripts/dev.sh own-cuda check-http 8015 "$OUT/http.json"
```

OUT 必须实际定义为一个新的绝对目录，模型子目录在调用前不存在；执行前确认端口可用，不终止用户其它进程。GPU sanitizer 按既有入口运行与最终改动相关的回归；没有硬件则写 BLOCKED，不把 CPU CI 当替代。

同时补齐删除/迁移所需的全库引用检查与文件规模统计。对尚未逐行阅读的文件记录覆盖范围，不声称全库无遗漏或零缺陷。

**Tests**：退出码、真实 token、SSE 唯一终态、停止回收、backend/layout/precision；本机身份、依赖/model hash 与 candidate CI。必要时运行既有 GPU fault/memcheck 检查。

**DoD**：每项有 PASS/FAIL/BLOCKED 与身份。旧 12528 比较、旧 profiler 和性能数据标为历史继承或独立实验，不写成本轮重跑。失败不追加 sweep，不降低门槛。

## T04 — README 收束 / P1

**Files**：README；承接被移内容的现有深层文档。

**Why**：读者先理解价值，而非先读所有命令和计划。

**Change**：主屏按 What、Why、四卖点、默认产品架构、少量结果、一个 trade-off、Quick Start、五份 Deep Dive 排列。CPU/CUDA/llama 三条路径区分。代码导航必须包含 MiniCudaRunner 和 CUDA Runtime。移走详细 API、Windows/reference backend 完整命令、流水账与旧计划导航。

图使用可维护的文本/Markdown/已有图资源即可，不引入图形构建依赖。主图不并排展示所有历史实验。

**Tests**：检查相对链接、路径、命令与 ASCII/图中数据流；人工完成 30 秒和 5 分钟阅读。

**DoD**：四卖点以内；读者可判断自研边界和默认配置；CPU 仍可运行，研究未被误写成默认。

## T05 — 性能入口 / P1

**Files**：`docs/PERFORMANCE.md`、README 结果摘要；历史 raw/summary 不修改。

**Why**：当前性能入口被 paging 研究占据。

**Change**：最多六行结果，覆盖 CPU micro、CUDA model、CUDA Serving、paging 同容量/同预算和 FP16 停止。逐行提供 workload、baseline、metric、limitation 与旧身份入口；把完整 paging 表留在研究页。

**Tests**：逐数与既有 JSON/analysis 对照；区分中位数之比和配对变化中位数；禁止把 SIMD 4.29× 写成端到端，把 +4.31% 写成低延迟或普适优势，把 14/24 写成全面胜出。

**DoD**：正结果、负结果和不确定性都有；不产生新性能实验；任何历史数字均未冒充当前主分支重新采集。

## T06 — 历史归档 / P1

**Files**：`docs/PROJECT_PLAN*`、`docs/NEXT_SPEC*`、`docs/NEXT_OPT_SPEC.md`；引用这些文件的 Markdown/脚本仅按实际扫描结果修改。

**Why**：项目已经冻结，旧计划不能继续像待执行指令。

**Change**：优先 `git mv` 到 `docs/history/plans/`，建立一个短索引，注明冻结日期。迁移前扫描 `git grep` 结果和相对锚点；冻结包内 source snapshot 不动。可能被历史复核器依赖的路径先保留或做明确兼容，不强行移动。

**Tests**：全部受影响链接和脚本引用有效；历史来源可追踪；现有测试不因移动路径失败。

**DoD**：README 无旧 feature roadmap 主入口；五份活跃文档足够理解当前系统；没有重复制一套历史正文。

## T07 — Default / Research / Internal 分层 / P1

**Files**：ARCHITECTURE、CUDA_SERVING、CUDA_RUNTIME、PRECISION_STUDY；必要的头文件注释。

**Why**：范围克制应体现为清楚的合同，而非模糊支持。

**Change**：明确 stable/default 为 CPU 快速路径和 F32/contiguous CUDA 展示路径；paged 为实验 opt-in；FP16 matrix 为 blocked research；debug logits、fault hooks、profiler timing 不属于服务承诺。说明 GPU stages 观测不提供虚假的逐阶段 GPU 计时。

强调 `BlockPool != GPU allocator`、`clear != cudaFree`、`CUDA preflight != CPU rollback`、`整批 sample 校验 != 整个网络事务`。

**Tests**：已有参数拒绝、capabilities 与 resources 测试；文档核对 config.cpp/MiniCudaRunner，不改变默认值。

**DoD**：不新增参数、接口或抽象；不把所有权类型为简化而合并；支持范围与源码一致。

## T08 — Evidence Hygiene / P1

**Files**：`docs/ARTIFACT_POLICY.md`、`benchmarks/results/cuda-vs-001/` 及实际受影响索引；其他实验包不随意重封。

**Why**：减少 Git 工作树的大型历史产物，同时保留唯一复现来源。

**Change**：对既有 56,872,322-byte ZIP 准备外部 canonical locator；实际无凭据下载并验证 SHA-256、文件完整性、既有复核器，再决定移除 Git 工作树副本。保留失败和 dirty source snapshot。

**Tests**：下载失败、hash 不符、缺文件时必须保留原件；旧研究阈值和报告字节不改。检查 CI 是否仍需要原路径。

**DoD**：每实验一个 canonical raw；迁移有实验证据。目标不可访问则 KEEP 并记录例外，不阻塞其它收尾。不得用历史重写来追求 clone size 数字。

## T09 — 分支清理 / P2

**Files**：分支清单/版本记录；远程 refs 仅在明确确认后变更。

**Why**：九个已整合的历史分支不再是活跃实施线。

**Change**：重新 fetch 后比较 ahead/behind、merge-base，检查本地未推送提交、未提交文件与活跃 PR。保留 main、研究 tag；删除仅限已确认整合的分支引用。

**Tests**：目标提交仍可从 main/tag 到达；无独有工作；Release 标签未移动。

**DoD**：无需合并的新代码不再被包装成 merge task；未获删除确认则输出清单即可，不影响作品交付。

## T10 — Portfolio Release Preparation / P1

**Files**：`CMakeLists.txt` 的版本行、VERSION_CONTROL 的当前发布说明、Release draft。

**Why**：研究发布不是整个可用系统的总览。

**Change**：建议 `v0.2.0`，先核未占用；发布基于最终 clean candidate，不复用 v0.1.0。正文含 supported scope、已验环境、architecture、3～6 行 results、limitations、negative results、reproduction 与已有 evidence locators。

**Tests**：exact SHA 的 CI 成功；WSL/硬件范围 smoke 实际完成；候选与记录的代码差异可解释；模型、凭据和本地状态不进入附件。

**DoD**：草稿可直接评审。用户未明确授权公开发布时不执行 publish/tag push；未完成本机验证时保持未发布或明确 draft，不虚报 ready。

## T11 — Resume / Demo / Interview Handoff / P1

**Files**：个人求职材料；公开仓库仅保留简短阅读和演示入口。

**Why**：代码存在不自动证明作者能解释设计和实验。

**Change**：四条简历，一主两副故事，十个必答问题。完整问题库留个人材料。演示复用 serve、check-http 和已有 benchmark；多请求实际重叠才声称演示了 continuous batching，查看 mixed_batches / max_batch_sequences，而不以“四次 curl”代替证据。

**Tests**：完成一次短生成、SSE、metrics 和正确停止；讲清 CPU/GPU/vendor 边界、KV 字节与故障完成点；所有简历数字可回到历史证据。

**DoD**：没有新 Web UI、演示框架或私造性能数字。现有脚本足够则不创建 demo.sh；不可确定调度重叠时说明未展示，而非改 scheduler 配合演示。

## T12 — Optional Polish / P2

**Files**：GitHub About、README 代码导航；可选 `src/minillm/gguf_model.cpp`。

**Why**：低风险改善首屏和所有权表达，不应扩大范围。

**Change**：About 简述 C++20 CPU/CUDA Runtime + Serving；可选将内部 MappedFile copy constructor/assignment 显式 delete，不增加接口层。没有实际复制调用，不宣称修复已复现 bug。

**Tests**：若改 C++，运行已有 CPU/core 测试及适用构建；纯元数据核对即可。

**DoD**：收益不明确可直接 SKIP，P2 不阻塞求职。

## 最终验收和禁止事项

最多十二项任务、三个阶段。P0 为当前公开事实和实际可复现性；P1 为展示与交付；P2 可跳过。

DO NOT TOUCH：Runtime math、scheduler 策略、HTTP 请求/响应完成语义、CUDA poison 合同、历史 benchmark 数值/阈值/原始身份、冻结输入、已发布 tags。

唯一允许的产品代码修复必须有确定断链的最小复现与定向回归；不得利用“修复”引入新 backend、模式或通用框架。若发现超出本规范的大问题，单独记录风险，不在收尾分支暗中重写系统。

提交前检查 `git diff --check`、显式暂存路径和文件大小；不 `git add .` 打包本地输出。新增脚本前置逻辑限约 80 行且只改现有入口。没有新的性能 trial、没有改门槛、没有新增 roadmap。

完成核心 P0/P1 后停止。三份收尾材料随后归档，不变成新的长期项目管理负担。
