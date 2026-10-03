#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
run_dir=${1:?run directory required}
sdk=/home/haneol/work/strata/.venv/lib/python3.12/site-packages/nvidia/cu13
exec >"$run_dir/build.log" 2>&1
record_exit() {
    rc=$?
    trap - EXIT
    jq -n --argjson exit_code "$rc" '{exit_code:$exit_code}' >"$run_dir/build-result.json"
    exit "$rc"
}
trap record_exit EXIT
if [[ -p "$run_dir/build-start.fifo" ]]; then
    IFS= read -r token <"$run_dir/build-start.fifo"
    [[ "$token" == start ]]
fi

cmake -S "$repo" -B "$repo/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MAKE_PROGRAM=/home/haneol/work/strata/.venv/bin/ninja \
    -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER="$sdk/bin/nvcc" \
    -DCUDAToolkit_ROOT="$sdk" -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_MMQ_KQUANTS=ON \
    -DSTRATA_GGML_DIR=/home/haneol/work/strata/third_party/llama.cpp
cmake --build "$repo/build" --target strata --parallel 2
sha256sum "$repo/build/strata" >"$run_dir/binary-sha256.txt"
ldd "$repo/build/strata" >"$run_dir/ldd.txt"
if rg -q 'not found' "$run_dir/ldd.txt"; then exit 1; fi
"$repo/build/strata" --help >"$run_dir/engine-help.txt" 2>&1
printf 'BUILD_COMPLETE\n'
