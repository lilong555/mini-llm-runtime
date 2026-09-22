# WSL2 原生开发

主工作区位于 Ubuntu 的 `/home/li/code/mini-llm-runtime`，源码、Git、依赖、模型和构建产物都存储在 Linux 文件系统。Windows 路径 `E:\code\LLM Serving Engine` 是独立副本，不与主工作区自动同步。

从 PowerShell 进入：

```powershell
wsl -d Ubuntu --cd /home/li/code/mini-llm-runtime
```

从 Windows 浏览文件使用 `\\wsl.localhost\Ubuntu\home\li\code\mini-llm-runtime`。编辑器使用 WSL 远程模式；终端、Git、CMake、编译器、调试器和服务均在 Ubuntu 中运行。避免在 `/mnt/e` 下构建或复用 Windows 的 CMake 缓存。

## CPU 工作流

依赖：Git、GCC 的 C++20 工具链、CMake >= 3.24、Ninja、Python 3。Python 仅用于模型文件管理，C++ 运行时不依赖 Python。

```bash
cd /home/li/code/mini-llm-runtime
bash scripts/dev.sh dependencies
bash scripts/dev.sh model
bash scripts/dev.sh build
bash scripts/dev.sh test
bash scripts/dev.sh validate
bash scripts/dev.sh check-http 8015
```

构建目录为 `build/wsl-cpu`，配置为 `RelWithDebInfo`，默认 4 个编译任务，可用 `JOBS=8 bash scripts/dev.sh build` 调整。模型文件依据 `models/manifest.json` 和 `models/reference-manifest.json` 校验大小及 SHA-256；已有不匹配文件导致失败，不会自动覆盖。

## 服务与 HTTP 验证

```bash
bash scripts/dev.sh serve --port 8000
```

在第二个 WSL 终端运行：

```bash
curl --fail http://127.0.0.1:8000/health
bash scripts/dev.sh http-test 8000
build/wsl-cpu/bin/llmserve-bench --port 8000 \
  --trace benchmarks/traces/cpu-mixed-s0.jsonl --output .run/wsl-replay.json
```

服务在前台运行，使用 Ctrl+C 退出；独立自动化进程可传入 `--shutdown-file .run/server.stop`，创建该文件请求停止。只监听 loopback，Windows 可通过 WSL localhost 转发访问。测试报告位于 `.run/wsl-model.json`、`.run/wsl-http.json` 和 `build/wsl-cpu/test-results.xml`。

PowerShell 策略编排及严格验收器仍属于独立开发中的 PLAN-001；Linux 的原始 replay 不等同于完整策略对照验收。未安装 `pwsh` 时 CTest 不注册 PowerShell fixture，应明确记录实际测试集合。

## CUDA 边界

WSL 中 `nvidia-smi` 可识别 GPU，不代表已安装 Linux CUDA Toolkit。Windows CUDA/MSVC 产物不能用于 Linux 构建。CUDA 参照后端需另行安装 Linux Toolkit、确认 `nvcc` 与架构，再在独立目录配置 `LLMSERVE_CUDA=ON`。当前原生开发入口使用 CPU；上游 GPU 执行不代表自研 CUDA PagedAttention。

## 工作区与版本管理

Linux 副本保留原分支、提交历史和未提交内容。后续开发以 Linux 副本为准，避免两侧交替修改同一分支。构建产物、服务状态、依赖 checkout 和模型权重不提交；保留模型来源与小型验证报告。Windows 副本保留用于恢复，不自动删除。
