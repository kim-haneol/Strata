#!/usr/bin/env bash
set -euo pipefail

run_dir=$1
repo_dir=$(cd "$(dirname "$0")/../.." && pwd)
sdk=/home/haneol/work/strata/.venv/lib/python3.12/site-packages/nvidia/cu13

exec >"$run_dir/build.log" 2>&1
record_exit() {
    exit_code=$?
    trap - EXIT
    jq -n --argjson exit_code "$exit_code" '{exit_code:$exit_code}' >"$run_dir/result.json"
    exit "$exit_code"
}
trap record_exit EXIT

IFS= read -r start_token <"$run_dir/start.fifo"
[[ "$start_token" == start ]]

cmake -S "$repo_dir" -B "$repo_dir/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MAKE_PROGRAM=/home/haneol/work/strata/.venv/bin/ninja \
    -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER="$sdk/bin/nvcc" \
    -DCUDAToolkit_ROOT="$sdk" -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_WERROR=ON \
    -DSTRATA_GGML_DIR=/home/haneol/work/strata/third_party/llama.cpp
cmake --build "$repo_dir/build" --target strata --parallel 6
sha256sum "$repo_dir/build/strata" >"$run_dir/binary-sha256.txt"
printf 'BUILD_COMPLETE\n'
