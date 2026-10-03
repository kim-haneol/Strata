#!/usr/bin/env bash
set -euo pipefail
mode=${1:?mode required}
run_dir=${2:?run directory required}
repo=$(cd "$(dirname "$0")/../.." && pwd)
exec >"$run_dir/bench.log" 2>&1
record_exit() {
    rc=$?
    trap - EXIT
    jq -n --argjson exit_code "$rc" '{exit_code:$exit_code}' >"$run_dir/bench-result.json"
    exit "$rc"
}
trap record_exit EXIT
IFS= read -r token <"$run_dir/bench-start.fifo"
[[ "$token" == start ]]
cd "$repo"
/home/haneol/.local/bin/uv run --no-project --python /home/haneol/work/strata/.venv/bin/python \
    python tools/local-ab/bench-api.py "$mode" "$run_dir" --port 22938
