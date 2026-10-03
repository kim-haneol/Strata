#!/usr/bin/env bash
set -euo pipefail

mode=${1:?layer-split or peer-host required}
run_dir=${2:?absolute run directory required}
repo=$(cd "$(dirname "$0")/../.." && pwd)
case "$mode" in layer-split|peer-host) ;; *) printf 'Unknown mode: %s\n' "$mode" >&2; exit 2;; esac
[[ "$run_dir" == /* ]]
mkdir -p "$run_dir"
command -v astop >/dev/null
astop --json >"$run_dir/preflight-astop.json"
uv run --no-project --python /home/haneol/work/strata/.venv/bin/python python - "$run_dir/preflight-astop.json" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1]))
gpus = [g for g in data['gpus'] if g['vendor_id'] == 4318]
if len(gpus) != 2:
    raise SystemExit('Expected two NVIDIA GPUs; refusing to load')
for g in gpus:
    memory = g.get('in_use_system_memory_bytes')
    utilization = g.get('device_utilization_percent')
    if memory is None or utilization is None:
        raise SystemExit('GPU telemetry unavailable; refusing to load')
    if memory > 512 * 1024**2 or utilization > 5:
        raise SystemExit(f"GPU {g['index']} is occupied; refusing to load")
PY
jq --arg log "$run_dir/engine.log" '.log=$log' "$repo/docs/local-ab/$mode.json" >"$run_dir/config.json"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
unit="strata-upstream-ab-$mode-$stamp"
systemd-run --user --unit="$unit" --property=Type=exec --property=WorkingDirectory="$repo" \
    --property="StandardOutput=append:$run_dir/server.log" --property="StandardError=append:$run_dir/server.log" \
    /home/haneol/.local/bin/uv run --no-project --python /home/haneol/work/strata/.venv/bin/python \
    python -m serve.server --engine strata --config "$run_dir/config.json" --host 127.0.0.1 --port 22938
pid=$(systemctl --user show "$unit.service" -p MainPID --value)
jq -n --arg unit "$unit.service" --arg watch "$unit" --argjson pid "$pid" --arg mode "$mode" \
    '{unit:$unit,watch:$watch,pid:$pid,mode:$mode,port:22938}' >"$run_dir/receipt.json"
astop wait --pid "$pid" --until start --timeout 10s --profile agent --json >"$run_dir/observed-start.json"
astop watch start --name "$unit" --pid "$pid" --log "$run_dir/server.log" \
    --owner codex --session strata-upstream-ab-20261003 --json >"$run_dir/watch.json"
cat "$run_dir/receipt.json"
