"""Freeze only the newly identified custom predictor/cache coupling controls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CUSTOM = Path("/home/haneol/work/strata-causal-custom")


def save(p, v):
    p.write_text(json.dumps(v, indent=2) + "\n")


def main():
    phase1 = Path(sys.argv[1]).resolve()
    assert json.loads((phase1 / "complete.json").read_text())["exit_code"] == 0
    original_plan = json.loads((phase1 / "plan.json").read_text())
    base = next(v for v in original_plan["variants"] if v["name"] == "C-base")
    old = json.loads((phase1 / "manifest.json").read_text())
    exe = CUSTOM / "build/strata-phase2"
    variants = []

    def variant(name, change=None, extra=None):
        a = list(base["args"])
        for key, value in (change or {}).items():
            a[a.index(key) + 1] = str(value)
        a += extra or []
        return dict(base, name=name, args=a, exe=str(exe))

    # Ordinary-generation gate checks the new opt-in flags' default against phase one's output.
    gate = variant("C-phase2-gate", {"--second-gpu-prefetch-mb": 0})
    for key in ("--spec-follow", "--diag-window-cap"):
        k = gate["args"].index(key)
        del gate["args"][k:k+2]
    gate.update(natural=True, sequence=["code"], expected_output_sha256=json.loads(
        (phase1 / "C-gate-patched/measurements.json").read_text())[0]["output_sha256"])
    variants += [
        variant("C-base-phase2"),
        variant("C-learn-static", extra=["--diag-predict-learn", "1"]),
        variant("C-adapt-unlearned", {"--adapt-every": 1}, ["--diag-predict-learn", "0"]),
        variant("C-adapt-prefetch0", {"--adapt-every": 1, "--second-gpu-prefetch-mb": 0}),
        variant("C-cache-match", extra=["--diag-second-slots", "2660"]),
        variant("C-base-phase2-repeat"),
    ]
    out = ROOT / "docs/causal-diag"
    save(out / "phase2-plan.json", {
        "status": "frozen_before_execution", "phase1_run": str(phase1),
        "fixtures": original_plan["fixtures"], "gates": [gate], "variants": variants,
        "reason": "Custom cache adaptation also enables predictor learning; separate both and check helper cache capacity.",
        "cache_match": "2460 primary + 2660 helper equals layer split's 5120 initial expert slots; placements/bytes still differ.",
        "limits": original_plan["limits"],
    })
    files = [x for x in old["files"] if not x["path"].endswith("/build/strata")]
    files += [{"path": str(exe), "bytes": exe.stat().st_size,
               "sha256": hashlib.sha256(exe.read_bytes()).hexdigest()}]
    save(out / "phase2-manifest.json", {"files": files, "custom_source_commit":
         subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CUSTOM, text=True).strip(),
         "phase1_binary": str(phase1 / "binaries/custom-strata"), "phase1_source_commits": old["source_commits"]})
    print("PHASE2_PREPARED lifetimes=7 requests=31")


if __name__ == "__main__":
    main()
