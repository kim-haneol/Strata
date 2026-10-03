"""Validate and summarize finite causal diagnostics without GPU work."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3


def save(p, data):
    p.write_text(json.dumps(data, indent=2) + "\n")


def median(rows, fixture, metric):
    a = [r[metric] for r in rows if r["fixture"] == fixture]
    return statistics.median(a) if a else None


def diagnostics(text):
    follows = re.findall(r"strata diagnostic follow: (\d+) of (\d+) emitted tokens differ from argmax; single=(\d+)", text)
    capacity = {}
    patterns = {
        "primary_filled": r"pre-filled (\d+) of (\d+) slots",
        "custom_helper": r"second GPU \d+ \([^)]+\): (\d+) experts from the profile and (\d+) empty slots, ([\d.]+) GiB",
        "peer_helper": r"peer GPU \d+: (\d+) experts, ([\d.]+) GiB",
        "layer_helper": r"layer split: CUDA\d+ runs layers [\d-]+, expert cache (\d+) slots \(([\d.]+) GiB\)",
        "prompt_loans": r"prompt path borrows.*",
        "chunk_warning": r"(?:WARNING: prompt chunk|reducing prompt chunk).*",
    }
    for key, pattern in patterns.items():
        capacity[key] = re.findall(pattern, text)
    return {"target_disagreements": [{"different": int(a), "emitted": int(b), "single": int(c)}
                                     for a, b, c in follows],
            "capacity": capacity, "timing_lines": [
                line for line in text.splitlines()
                if "diagnostic timing:" in line or "decode timing:" in line or "decode GPU stages" in line
                or re.search(r"strata serve: stage \d", line)]}


def telemetry(path, rows, run_dir):
    gpus = {}
    pressures = set()
    peak_ram = peak_cpu = 0
    min_swap = max_swap = None
    samples = 0
    intervals = [(r["end_unix"] - r["decode_ms"] / 1000, r["end_unix"], r) for r in rows]
    pids = {r["variant"]: json.loads((run_dir / r["variant"] / "receipt.json").read_text())["pid"] for r in rows}
    aligned = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            samples += 1
            stamp = datetime.fromisoformat(d["timestamp_utc"].replace("Z", "+00:00")).timestamp()
            m = d["memory"]
            peak_ram = max(peak_ram, m["used_bytes"])
            pressures.add(m["pressure"])
            sw = m["swap_used_bytes"]
            min_swap = sw if min_swap is None else min(min_swap, sw)
            max_swap = sw if max_swap is None else max(max_swap, sw)
            cpu = d["cpu"].get("utilization_percent")
            if cpu is not None:
                peak_cpu = max(peak_cpu, cpu)
            for g in d["gpus"]:
                if g["vendor_id"] != 4318:
                    continue
                v = gpus.setdefault(str(g["index"]), {})
                for field in ("device_utilization_percent", "in_use_system_memory_bytes",
                              "power_watts", "frequency_mhz", "temperature_c"):
                    x = g.get(field)
                    if x is not None:
                        v[field] = max(v.get(field, x), x)
            for start, end, row in intervals:
                if not start <= stamp <= end:
                    continue
                key = row["variant"] + "/" + row["fixture"]
                a = aligned.setdefault(key, {"samples": 0, "gpu": {}, "cpu": [], "per_core_peak": [],
                                               "process_cpu": [], "process_rss": [], "process_gpu": [], "process_vram": [],
                                               "openroad_cpu": []})
                a["samples"] += 1
                a["cpu"].append(cpu)
                a["per_core_peak"].append(max(d["cpu"]["per_core_percent"]))
                a["openroad_cpu"].append(sum(p["cpu_percent"] or 0 for p in d["processes"] if p["name"] == "openroad"))
                for g in d["gpus"]:
                    if g["vendor_id"] == 4318:
                        a["gpu"].setdefault(str(g["index"]), []).append(g["device_utilization_percent"])
                process = next((p for p in d["processes"] if p["pid"] == pids[row["variant"]]), None)
                if process:
                    for key2, field in (("process_cpu", "cpu_percent"), ("process_rss", "mem_resident_bytes"),
                                        ("process_gpu", "gpu_percent"), ("process_vram", "gpu_mem_bytes")):
                        a[key2].append(process.get(field))
    def measured(values, reducer):
        known = [x for x in values if x is not None]
        return reducer(known) if known else None
    compact = {key: {
        "samples": a["samples"], "device_utilization_mean_percent": {g: measured(v, statistics.mean) for g, v in a["gpu"].items()},
        "cpu_mean_percent": measured(a["cpu"], statistics.mean), "per_core_peak_percent": measured(a["per_core_peak"], max),
        "process_cpu_mean_percent": measured(a["process_cpu"], statistics.mean),
        "process_rss_peak_bytes": measured(a["process_rss"], max),
        "process_gpu_mean_percent": measured(a["process_gpu"], statistics.mean),
        "process_vram_peak_bytes": measured(a["process_vram"], max),
        "openroad_cpu_mean_percent": measured(a["openroad_cpu"], statistics.mean),
        "openroad_cpu_peak_percent": measured(a["openroad_cpu"], max),
    } for key, a in aligned.items()}
    return {"samples": samples, "peak_used_ram_gib": peak_ram / GIB, "pressure_states": sorted(pressures),
            "swap_min_gib": min_swap / GIB, "swap_max_gib": max_swap / GIB, "peak_cpu_percent": peak_cpu,
            "gpu_peaks": gpus, "approximate_decode_intervals": compact}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--partial", action="store_true")
    a = p.parse_args()
    plan = json.loads((a.run_dir / "plan.json").read_text())
    expected = plan["gates"] + plan["variants"]
    if not a.partial:
        complete = json.loads((a.run_dir / "complete.json").read_text())
        assert complete["exit_code"] == 0 and complete["variants"] == len(expected)
    result = {"run_dir": str(a.run_dir.resolve()), "limits": plan["limits"], "variants": {}}
    rows_all = []
    for v in expected:
        d = a.run_dir / v["name"]
        if not (d / "exit.json").exists():
            if a.partial:
                continue
            raise RuntimeError(f"Incomplete engine lifetime: {v['name']}")
        ex = json.loads((d / "exit.json").read_text())
        assert ex["exit_code"] == 0
        rows = json.loads((d / "measurements.json").read_text())
        assert len(rows) == len(v["sequence"])
        if not v.get("natural"):
            assert all(r["generated"] == 300 and r["reused"] == 0 for r in rows)
            assert all(r["output_sha256"] == plan["fixtures"]["follow"]["sha256"] for r in rows)
        rows_all += rows
        summary = {key: median(rows, fixture, metric) for key, fixture, metric in (
            ("code_decode_tok_s", "code", "decode_tok_s"), ("edit_decode_tok_s", "edit", "decode_tok_s"),
            ("long_prefill_tok_s", "long-code", "prefill_tok_s"), ("long_decode_tok_s", "long-code", "decode_tok_s"))}
        summary.update(diagnostics((d / "engine.log").read_text()))
        summary.update(effect=v["effect"], measurements=rows, exit=ex)
        result["variants"][v["name"]] = summary
    for mode in ("C", "P"):
        names = [f"{mode}-gate-{kind}" for kind in ("original", "patched")]
        if all(n in result["variants"] for n in names):
            hashes = [result["variants"][n]["measurements"][0]["output_sha256"] for n in names]
            assert hashes[0] == hashes[1], f"Diagnostic-off behavior gate {mode} failed"
    bases = {}
    for mode in ("C", "P", "L"):
        names = [n for n in (mode + "-base", mode + "-base-repeat") if n in result["variants"]]
        if names:
            bases[mode] = {metric: statistics.median([result["variants"][n][metric] for n in names])
                           for metric in ("code_decode_tok_s", "edit_decode_tok_s", "long_prefill_tok_s", "long_decode_tok_s")}
    result["reference_centers"] = bases
    result["request_count"] = len(rows_all)
    if not a.partial:
        assert len(rows_all) == 102
        result["telemetry"] = telemetry(a.run_dir / "telemetry.jsonl", rows_all, a.run_dir)
        for mode in ("C", "P", "L"):
            reference = result["variants"][mode + "-base"]["capacity"]
            for suffix in ("single", "lookup", "adapt", "base-repeat"):
                capacity = result["variants"][mode + "-" + suffix]["capacity"]
                for key in ("primary_filled", "custom_helper", "peer_helper", "layer_helper"):
                    assert capacity[key] == reference[key], (mode, suffix, key, capacity[key], reference[key])
        result["matched_capacity_checks_passed"] = True
    for name, v in result["variants"].items():
        if "-gate-" in name or "profile" in name:
            continue
        v["change_vs_reference_percent"] = {
            metric: 100 * (v[metric] / bases[name[0]][metric] - 1)
            for metric in bases[name[0]] if v[metric] is not None}
    out = ROOT / "docs/causal-diag"
    destination = "partial-results.json" if a.partial else "results.json"
    save(out / destination, result)
    if a.partial:
        print(json.dumps({"completed": list(result["variants"]), "reference_centers": bases}))
        return
    lines = ["# Controlled diagnostic results", "", "Date: 2026-10-03 UTC.",
             "C = custom helper, P = upstream host peer, L = upstream layer split.",
             "All 26 engine lifetimes returned exit 0. Four ordinary-generation gates passed;",
             "98 diagnostic responses emitted the same 300-token reference with no prompt reuse.", "",
             "| Condition | Code median tok/s | Edit tok/s | 32K prefill tok/s | Decode after 32K tok/s |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, v in result["variants"].items():
        if "-gate-" in name or "profile" in name:
            continue
        values = [v[k] for k in ("code_decode_tok_s", "edit_decode_tok_s", "long_prefill_tok_s", "long_decode_tok_s")]
        lines.append("| " + name + " | " + " | ".join(f"{x:.1f}" for x in values) + " |")
    lines += ["", "The code median uses three consecutive requests in one fresh process.",
              "Closing references are independent reloads; all other interventions have one lifetime.",
              "Baseline centers average the two reference medians, exposing but not eliminating drift.",
              "These are conditional effects. Do not add them as shares of the historical gap.", "",
              "## Interpretation limits", ""]
    lines += [f"- {x}" for x in plan["limits"]]
    t = result["telemetry"]
    lines += ["", "## Observation and completion", "",
              f"Native ASTOP recording: {t['samples']} samples; peak used RAM {t['peak_used_ram_gib']:.2f} GiB;",
              f"memory pressure states {', '.join(t['pressure_states'])}; swap {t['swap_min_gib']:.2f}-{t['swap_max_gib']:.2f} GiB.",
              "GPU peaks and request-aligned process/CPU/per-core evidence remain in the recording.",
              "GPU utilization includes possible wait kernels; it is not kernel compute time.",
              "Direct child exit 0 and exact response validation are completion authority;",
              "ASTOP observed exit status can be unavailable and is retained as unknown.",
              "", "Raw evidence: " + str(a.run_dir.resolve()),
              "Frozen hashes: manifest.json. Full measurements and diagnostic timing lines: results.json.",
              "Initial CPU builds and a launcher/EOF cleanup recovery are preserved in earlier run directories;",
              "those attempts are excluded from the completed experiment."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"requests": len(rows_all), "reference_centers": bases,
                      "telemetry_peaks": {k: v for k, v in t.items() if k != "approximate_decode_intervals"}}))


if __name__ == "__main__":
    main()
