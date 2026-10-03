#!/usr/bin/env bash
set -euo pipefail

run_dir=$1
repo_dir=$(cd "$(dirname "$0")/../.." && pwd)
model=/home/haneol/models/hf/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
pack=/home/haneol/models/strata/packs/ud-q4-k-xl-custom
python=/home/haneol/work/strata/.venv/bin/python

exec >"$run_dir/prepare.log" 2>&1
record_exit() {
    exit_code=$?
    trap - EXIT
    jq -n --argjson exit_code "$exit_code" '{exit_code:$exit_code}' >"$run_dir/result.json"
    exit "$exit_code"
}
trap record_exit EXIT

IFS= read -r start_token <"$run_dir/start.fifo"
[[ "$start_token" == start ]]

cd "$repo_dir"
/home/haneol/.local/bin/uv run --no-project --python "$python" python tools/iq_pack.py --gguf "$model" --out "$pack"
find "$pack" -maxdepth 2 -type f -printf '%P\t%s\n' | sort >"$run_dir/pack-files.txt"
sha256sum "$pack/index.txt" "$pack/native_experts.txt" "$pack/dense.bin" >"$run_dir/pack-sha256.txt"
printf 'PACK_COMPLETE\n'
