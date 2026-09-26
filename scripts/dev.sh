#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
build=build/wsl-cpu
model=models/Qwen3-0.6B-Q8_0.gguf
backend=mini
gpu_layers=0
cuda=OFF
own_cuda=OFF
report_prefix=.run/wsl
if [[ ${1:-} == cuda ]]; then
  shift
  build=build/wsl-cuda
  backend=llama
  gpu_layers=99
  cuda=ON
  report_prefix=.run/wsl-cuda
elif [[ ${1:-} == own-cuda ]]; then
  shift
  build=build/wsl-own-cuda
  own_cuda=ON
  case "${1:-help}" in
    build|test|memcheck|storage-check|storage-memcheck|layer-check|layer-memcheck|model-check|model-full-check|model-memcheck|runtime-benchmark|generate|help) ;;
    *) echo '自有 CUDA 提供 build/test/memcheck、storage/layer/model 验证、runtime-benchmark 和 generate；GPU Serving 尚未交付。' >&2; exit 1 ;;
  esac
fi
case "${1:-help}" in
  build)
    options=(-DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLMSERVE_WITH_LLAMA=ON
      -DLLMSERVE_BUILD_TESTS=ON "-DLLMSERVE_CUDA=$cuda" "-DMINILLM_ENABLE_CUDA=$own_cuda"
      -DLLMSERVE_REQUIRE_TEST_TOOLS=ON)
    if [[ $cuda == ON || $own_cuda == ON ]]; then
      options+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES:-89}")
    fi
    cmake -S . -B "$build" -G Ninja "${options[@]}"
    cmake --build "$build" --parallel "${JOBS:-4}"
    ;;
  test)
    ctest --test-dir "$build" --output-on-failure --no-tests=error \
      --test-output-size-passed 65536 --output-junit test-results.xml
    ;;
  memcheck)
    [[ $own_cuda == ON ]] || { echo '用法：bash scripts/dev.sh own-cuda memcheck' >&2; exit 1; }
    compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 "$build/bin/minillm-cuda-unit-tests"
    compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 "$build/bin/minillm-cuda-ops-tests"
    compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 "$build/bin/minillm-cuda-storage-tests"
    compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 "$build/bin/minillm-cuda-layer-tests"
    compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 "$build/bin/minillm-cuda-runtime-tests"
    ;;
  generate)
    [[ $own_cuda == ON ]] || { echo '用法：bash scripts/dev.sh own-cuda generate [生成参数]' >&2; exit 1; }
    shift
    exec "$build/bin/mini-cuda-llm" --model "$model" "$@"
    ;;
  model-check|model-full-check|model-memcheck)
    [[ $own_cuda == ON && $# -le 2 ]] || { echo '用法：bash scripts/dev.sh own-cuda model-check [新报告目录]' >&2; exit 1; }
    model_output=${2:-".run/cuda-model-$(date -u +%Y%m%dT%H%M%S)-$$"}
    model_command=("$build/bin/minillm-cuda-model-tests" --model "$model"
      --reference-model models/Qwen3-0.6B-Q8_0-dequant-F32.gguf
      --contract tests/data/qwen3_validation_cases.json --output "$model_output")
    if [[ $1 == model-full-check ]]; then model_command+=(--full); fi
    if [[ $1 == model-memcheck ]]; then
      compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 "${model_command[@]}"
    else
      "${model_command[@]}"
    fi
    ;;
  layer-check|layer-memcheck)
    [[ $own_cuda == ON && $# -le 2 ]] || { echo '用法：bash scripts/dev.sh own-cuda layer-check [新报告目录]' >&2; exit 1; }
    layer_output=${2:-".run/cuda-layer-$(date -u +%Y%m%dT%H%M%S)-$$"}
    layer_command=("$build/bin/minillm-cuda-layer-tests" --model "$model"
      --contract tests/data/qwen3_validation_cases.json --output "$layer_output")
    if [[ $1 == layer-memcheck ]]; then
      compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 "${layer_command[@]}"
    else
      "${layer_command[@]}"
    fi
    ;;
  storage-check|storage-memcheck)
    [[ $own_cuda == ON && $# -le 2 ]] || { echo '用法：bash scripts/dev.sh own-cuda storage-check [新报告目录]' >&2; exit 1; }
    storage_output=${2:-".run/cuda-storage-$(date -u +%Y%m%dT%H%M%S)-$$"}
    storage_command=("$build/bin/minillm-cuda-storage-tests" --model "$model"
      --contract tests/data/qwen3_validation_cases.json --output "$storage_output")
    if [[ $1 == storage-memcheck ]]; then
      compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 "${storage_command[@]}"
    else
      "${storage_command[@]}"
    fi
    ;;
  model)
    python3 scripts/models.py
    ;;
  validate)
    python3 scripts/models.py --reference --converter "$build/bin/mini-llm"
    mkdir -p .run
    "$build/bin/llmserve-model-tests" --model "$model" \
      --reference-model models/Qwen3-0.6B-Q8_0-dequant-F32.gguf \
      --gpu-layers "$gpu_layers" --output "$report_prefix-model.json"
    ;;
  serve)
    shift
    exec "$build/bin/llmserve" --model "$model" --backend "$backend" \
      --gpu-layers "$gpu_layers" "$@"
    ;;
  http-test)
    mkdir -p .run
    "$build/bin/llmserve-http-tests" --port "${2:-8000}" --output "$report_prefix-http.json"
    ;;
  benchmark)
    shift
    exec pwsh -NoProfile -File scripts/Benchmark-Policies.ps1 \
      -Backend "$backend" -BinaryDirectory "$build/bin" "$@"
    ;;
  runtime-benchmark)
    shift
    if [[ $own_cuda == ON ]]; then
      exec pwsh -NoProfile -File scripts/Benchmark-CudaRuntime.ps1 -BinaryDirectory "$build/bin" "$@"
    else
      exec pwsh -NoProfile -File scripts/Benchmark-Runtime.ps1 -BinaryDirectory "$build/bin" "$@"
    fi
    ;;
  check-http)
    mkdir -p .run
    state=$(mktemp -d "$root/.run/http-check.XXXXXX")
    port=${2:-8015}
    "$build/bin/llmserve" --model "$model" --backend "$backend" --gpu-layers "$gpu_layers" \
      --port "$port" --shutdown-file "$state/stop" >"$state/server.log" 2>&1 &
    server_pid=$!
    trap 'touch "$state/stop"; wait "$server_pid" || true' EXIT
    ready=false
    for ((attempt=0; attempt<150; ++attempt)); do
      kill -0 "$server_pid" 2>/dev/null || { cat "$state/server.log"; exit 1; }
      if curl --fail --silent "http://127.0.0.1:$port/health" >/dev/null; then
        ready=true
        break
      fi
      sleep 0.2
    done
    [[ $ready == true ]] || { echo '服务就绪检查超时' >&2; exit 1; }
    "$build/bin/llmserve-http-tests" --port "$port" --output "$report_prefix-http.json"
    ;;
  smoke)
    [[ $cuda == ON ]] || { echo '用法：bash scripts/dev.sh cuda smoke' >&2; exit 1; }
    architecture=${CUDA_ARCHITECTURES:-89}
    [[ $architecture =~ ^[0-9]+$ ]] || { echo '冒烟检查只接受单个数字架构，如 89' >&2; exit 1; }
    mkdir -p .run
    nvcc -std=c++17 -O2 -lineinfo "-arch=sm_$architecture" \
      scripts/cuda_smoke.cu -o .run/cuda-smoke
    .run/cuda-smoke
    ;;
  dependencies)
    target=third_party/llama.cpp
    commit=911f6cdc8ab8a530b2bee09ee61471a6f3178eeb
    if [[ ! -e "$target" ]]; then
      git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git "$target"
      git -C "$target" checkout --detach "$commit"
    fi
    [[ $(git -C "$target" rev-parse HEAD) == "$commit" ]] || { echo 'Dependency revision mismatch' >&2; exit 1; }
    [[ -z $(git -C "$target" status --porcelain) ]] || { echo 'Dependency has local changes' >&2; exit 1; }
    ;;
  *)
    echo '用法：bash scripts/dev.sh [cuda] {dependencies|model|build|test|validate|serve [服务参数]|http-test [端口]|check-http [端口]|benchmark -Trace 路径 [基准参数]|runtime-benchmark [基准参数]|smoke}'
    echo '自有 CUDA：bash scripts/dev.sh own-cuda {build|test|memcheck|storage-check [新报告目录]|storage-memcheck [新报告目录]|layer-check [新报告目录]|layer-memcheck [新报告目录]|model-check [新报告目录]|model-full-check [新报告目录]|model-memcheck [新报告目录]|runtime-benchmark [基准参数]|generate [生成参数]}'
    [[ ${1:-help} == help ]]
    ;;
esac
