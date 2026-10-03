#!/usr/bin/env bash
set -euo pipefail
sdk=/home/haneol/work/strata/.venv/lib/python3.12/site-packages/nvidia/cu13
run_dir=/home/haneol/work/strata-causal-upstream/local-runs/causal-build-20261003
record_exit() {
    rc=$?
    trap - EXIT
    jq -n --argjson exit_code "$rc" '{exit_code:$exit_code}' >"$run_dir/exit.json"
    exit "$rc"
}
trap record_exit EXIT
for repo in /home/haneol/work/strata-causal-upstream /home/haneol/work/strata-causal-custom; do
    cmake -S "$repo" -B "$repo/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_MAKE_PROGRAM=/home/haneol/work/strata/.venv/bin/ninja \
        -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER="$sdk/bin/nvcc" \
        -DCUDAToolkit_ROOT="$sdk" -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
        -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_GGML_DIR=/home/haneol/work/strata/third_party/llama.cpp
    cmake --build "$repo/build" --target strata --parallel 4
    sha256sum "$repo/build/strata"
done
for repo in /home/haneol/work/strata-causal-upstream /home/haneol/work/strata-causal-custom; do
    cmake --build "$repo/build" --target strata --parallel 4
    sha256sum "$repo/build/strata"
done
printf 'CAUSAL_BUILD_COMPLETE\n'
