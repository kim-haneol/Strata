#!/usr/bin/env bash
set -euo pipefail
repo=/home/haneol/work/strata-causal-custom
evidence=/home/haneol/work/strata-causal-upstream/local-runs/causal-20261003T095910Z
record_exit() {
    rc=$?
    trap - EXIT
    jq -n --argjson exit_code "$rc" '{exit_code:$exit_code}' >"$evidence/phase2-build-exit.json"
    exit "$rc"
}
trap record_exit EXIT
cmake --build "$repo/build" --target strata --parallel 2
mv "$repo/build/strata" "$repo/build/strata-phase2"
cp --reflink=auto --preserve=timestamps "$evidence/binaries/custom-strata" "$repo/build/strata"
sha256sum "$repo/build/strata-phase2" "$repo/build/strata"
printf 'PHASE2_BUILD_COMPLETE\n'
