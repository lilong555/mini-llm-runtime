#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
build=build/wsl-cpu
model=models/Qwen3-0.6B-Q8_0.gguf
case "${1:-help}" in
  build)
    cmake -S . -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLMSERVE_CUDA=OFF
    cmake --build "$build" --parallel "${JOBS:-4}"
    ;;
  test)
    ctest --test-dir "$build" --output-on-failure --output-junit test-results.xml
    ;;
  model)
    python3 scripts/models.py
    ;;
  validate)
    python3 scripts/models.py --reference
    mkdir -p .run
    "$build/bin/llmserve-model-tests" --model "$model" \
      --reference-model models/Qwen3-0.6B-Q8_0-dequant-F32.gguf \
      --gpu-layers 0 --output .run/wsl-model.json
    ;;
  serve)
    shift
    exec "$build/bin/llmserve" --model "$model" --backend mini --gpu-layers 0 "$@"
    ;;
  http-test)
    mkdir -p .run
    "$build/bin/llmserve-http-tests" --port "${2:-8000}" --output .run/wsl-http.json
    ;;
  check-http)
    mkdir -p .run
    state=$(mktemp -d "$root/.run/http-check.XXXXXX")
    port=${2:-8015}
    "$build/bin/llmserve" --model "$model" --backend mini --gpu-layers 0 \
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
    [[ $ready == true ]] || { echo 'Server readiness timed out' >&2; exit 1; }
    "$build/bin/llmserve-http-tests" --port "$port" --output .run/wsl-http.json
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
    echo 'Usage: bash scripts/dev.sh {dependencies|model|build|test|validate|serve [server options]|http-test [port]|check-http [port]}'
    [[ ${1:-help} == help ]]
    ;;
esac
