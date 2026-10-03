#!/usr/bin/env bash
set -euo pipefail

run_dir=$1
config=$2
prompt=$3
follow=$4
shift 4

exec >"$run_dir/profile.log" 2>&1
monitor_pid=
record_exit() {
    exit_code=$?
    trap - EXIT
    if [[ -n $monitor_pid ]]; then
        kill -TERM "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    jq -n --argjson exit_code "$exit_code" '{exit_code:$exit_code}' >"$run_dir/result.json"
    exit "$exit_code"
}
trap record_exit EXIT

IFS= read -r start_token <"$run_dir/start.fifo"
[[ "$start_token" == start ]]

mapfile -t engine_args < <(jq -r '.args[]' "$config")
mapfile -t gpu_order < <(jq -r '.gpu[]' "$config")
mapfile -t lib_dirs < <(jq -r '.lib_dirs[]' "$config")
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(IFS=,; printf '%s' "${gpu_order[*]}")
export LD_LIBRARY_PATH
LD_LIBRARY_PATH=$(IFS=:; printf '%s' "${lib_dirs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
max_new=$(wc -w <"$follow")

astop --record "$run_dir/telemetry.jsonl" --interval 1s >"$run_dir/telemetry.log" 2>&1 &
monitor_pid=$!
jq -n --argjson max_new "$max_new" --arg cuda_visible "$CUDA_VISIBLE_DEVICES" --argjson extra "$(printf '%s\n' "$@" | jq -R . | jq -s .)" \
    '{max_new:$max_new,cuda_visible_devices:$cuda_visible,extra_args:$extra}' >"$run_dir/profile-plan.json"

"$(jq -r '.exe' "$config")" "${engine_args[@]}" --tokens-file "$prompt" --max-new "$max_new" \
    --spec-follow "$follow" --stats --window-profile "$@"
printf 'PROFILE_COMPLETE\n'
