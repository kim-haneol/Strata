"""Validate separated controls and write an evidence-backed combined interpretation."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

import report

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/causal-diag"
METRICS = ("code_decode_tok_s", "edit_decode_tok_s", "long_prefill_tok_s", "long_decode_tok_s")


def main():
    raw = Path(sys.argv[1]).resolve()
    plan = json.loads((raw / "plan.json").read_text())
    complete = json.loads((raw / "complete.json").read_text())
    assert complete == {"exit_code": 0, "variants": 7, "requests": 31}
    result = {"run_dir": str(raw), "variants": {}, "limits": plan["limits"]}
    all_rows = []
    for v in plan["gates"] + plan["variants"]:
        d = raw / v["name"]
        ex = json.loads((d / "exit.json").read_text())
        assert ex["exit_code"] == 0
        rows = json.loads((d / "measurements.json").read_text())
        assert len(rows) == len(v["sequence"])
        if v.get("natural"):
            assert rows[0]["output_sha256"] == v["expected_output_sha256"]
        else:
            assert all(r["generated"] == 300 and r["reused"] == 0 and
                       r["output_sha256"] == plan["fixtures"]["follow"]["sha256"] for r in rows)
        summary = {key: report.median(rows, fixture, metric) for key, fixture, metric in (
            ("code_decode_tok_s", "code", "decode_tok_s"), ("edit_decode_tok_s", "edit", "decode_tok_s"),
            ("long_prefill_tok_s", "long-code", "prefill_tok_s"), ("long_decode_tok_s", "long-code", "decode_tok_s"))}
        summary.update(report.diagnostics((d / "engine.log").read_text()), measurements=rows, exit=ex)
        result["variants"][v["name"]] = summary
        all_rows += rows
    assert len(all_rows) == 31
    ref = result["variants"]["C-base-phase2"]["capacity"]
    for v in plan["variants"]:
        cap = result["variants"][v["name"]]["capacity"]
        assert cap["primary_filled"] == ref["primary_filled"]
        if v["name"] == "C-cache-match":
            assert int(cap["primary_filled"][0][0]) == 2460
            assert int(cap["custom_helper"][0][0]) == 2660
        else:
            assert cap["custom_helper"] == ref["custom_helper"]
    centers = {m: statistics.median([result["variants"][n][m]
               for n in ("C-base-phase2", "C-base-phase2-repeat")]) for m in METRICS}
    result["reference_center"] = centers
    result["telemetry"] = report.telemetry(raw / "telemetry.jsonl", all_rows, raw)
    report.save(OUT / "phase2-results.json", result)
    first = json.loads((OUT / "results.json").read_text())
    c1 = first["reference_centers"]["C"]
    a = first["variants"]
    b = result["variants"]
    def change(value, reference):
        return 100 * (value / reference - 1)
    lines = [
        "# Findings from the two-GPU diagnostic experiment",
        "",
        "The custom fork's large observed prefill advantage uses the helper GPU's prompt processing and",
        "the larger prompt chunk. Its short-code throughput also benefits strongly from adaptive expert",
        "residency. MTP is active and useful in both upstream modes; it does not explain the gap through",
        "its presence alone. The measured effects below are conditional on these fixtures and settings.",
        "",
        "**CPU concurrency:** independent OpenROAD jobs ran during the experiment. Native ASTOP observed",
        "variable background CPU use (roughly 218-800% aggregate process CPU in the first reference/code intervals).",
        "The parameter interventions hold engine inputs and settings as described, but do not isolate the host",
        "from those jobs. This limits exact causal attribution, especially for small throughput differences.",
        "Closing references and process/per-core observations are retained; no confidence interval is claimed.",
        "",
        "There were 33 fresh engine lifetimes and 133 validated responses: five ordinary-generation",
        "behavior gates and 128 fixed 300-token diagnostic responses. Every engine returned exit 0.",
        "Diagnostic-off generation matched the corresponding original/first-phase token sequence.",
        "",
        "## Prefill and speculation",
        "",
        "| Intervention | Reference | Changed condition | Observed difference |",
        "| --- | ---: | ---: | ---: |",
        f"| Custom 32K prompt, 8K to 16K chunk | {c1['long_prefill_tok_s']:.1f} tok/s | {a['C-chunk16K']['long_prefill_tok_s']:.1f} tok/s | {change(a['C-chunk16K']['long_prefill_tok_s'],c1['long_prefill_tok_s']):+.1f}% |",
        f"| Custom 32K prompt, helper off to on | {a['C-prompt-main']['long_prefill_tok_s']:.1f} tok/s | {c1['long_prefill_tok_s']:.1f} tok/s | {change(c1['long_prefill_tok_s'],a['C-prompt-main']['long_prefill_tok_s']):+.1f}% |",
        "",
        "The chunk and helper effects are separate interventions at the stated reference settings.",
        "Do not add their percentages. Turning off helper prompt processing also changes downstream numerical",
        "state; its post-prompt decode is retained separately in the data.",
        "",
        "| Fixed-text code decode | One-token windows, MTP memory retained | MTP windows |",
        "| --- | ---: | ---: |",
    ]
    for mode, label in (("C", "Custom"), ("P", "Upstream peer"), ("L", "Upstream layer split")):
        lines.append(f"| {label} | {a[mode+'-single']['code_decode_tok_s']:.1f} | {first['reference_centers'][mode]['code_decode_tok_s']:.1f} |")
    lines += [
        "",
        "All three reference modes emitted about 3.4 tokens per verification window on the short code.",
        "Draft time was about 1.9-2.2 ms/window. The larger reference differences lay in verification",
        "and host/CPU-pool work. These timer categories can overlap and are not additive contributions.",
        "Single-token mode retains drafter VRAM, scratch capacity and prompt computation; it isolates",
        "decode speculation rather than the memory savings of unloading MTP.",
        "",
        "## Separating adaptation, prediction and cache count",
        "",
        "Custom normally ties predictor learning to cache adaptation. Explicit diagnostic flags separated",
        "the two. The second phase keeps the same main/cache budgets and changes only the specified control.",
        "",
        "| Second-phase custom condition | Code median tok/s | Edit tok/s | Decode after 32K tok/s |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, v in b.items():
        if name.endswith("gate"):
            continue
        lines.append(f"| {name} | {v['code_decode_tok_s']:.1f} | {v['edit_decode_tok_s']:.1f} | {v['long_decode_tok_s']:.1f} |")
    lines += [
        "",
        f"Second-phase reference center: {centers['code_decode_tok_s']:.1f} tok/s code.",
        f"First-phase adaptation plus learning: {a['C-adapt']['code_decode_tok_s']:.1f} tok/s code.",
        f"Adaptation without predictor learning reached {b['C-adapt-unlearned']['code_decode_tok_s']:.1f} tok/s",
        f"({change(b['C-adapt-unlearned']['code_decode_tok_s'],centers['code_decode_tok_s']):+.1f}%).",
        f"Learning alone reached {b['C-learn-static']['code_decode_tok_s']:.1f} tok/s",
        f"({change(b['C-learn-static']['code_decode_tok_s'],centers['code_decode_tok_s']):+.1f}%).",
        f"The two reference medians were {b['C-base-phase2']['code_decode_tok_s']:.1f} and",
        f"{b['C-base-phase2-repeat']['code_decode_tok_s']:.1f} tok/s (about 7% apart).",
        "The learning-only difference is inside that reference variation, so no reliable independent",
        "short-code gain is established. The much larger adaptation-only difference persists.",
        "Prefetch disabled under adaptation reached 99.4 tok/s, but the full adaptation+learning",
        "reference is in the first phase: do not attribute the entire cross-phase difference to prefetch.",
        "",
        "The expert-count control starts with exactly 2460 main + 2660 helper experts, matching",
        "layer split's 5120 total. Its byte footprint, placement and execution architecture still differ.",
        "It tests a conditional capacity reduction inside custom; it is not a pure kernel comparison.",
        f"The count reduction's code median was {b['C-cache-match']['code_decode_tok_s']:.1f} tok/s,",
        "inside the reference variation. Post-32K decode and prefill fell by",
        f"{-change(b['C-cache-match']['long_decode_tok_s'],centers['long_decode_tok_s']):.1f}% and",
        f"{-change(b['C-cache-match']['long_prefill_tok_s'],centers['long_prefill_tok_s']):.1f}%, respectively.",
        "",
        "## Lookup depends on the request",
        "",
        f"Custom lookup changed short-code median {c1['code_decode_tok_s']:.1f} -> {a['C-lookup']['code_decode_tok_s']:.1f} tok/s,",
        f"but the request containing the reference code changed {c1['edit_decode_tok_s']:.1f} -> {a['C-lookup']['edit_decode_tok_s']:.1f} tok/s.",
        "Upstream's small short-code lookup changes are comparable to its reference variation; the edit",
        "case has a larger gain. Peer hot-placement changes also traded short-code speed against",
        "post-32K speed. These results do not establish one best configuration for every workload.",
        "",
        "## Evidence and limits",
        "",
        "The continuation was saved from a historical custom response. Forced tokens control text and",
        "length but do not guarantee identical routing or numerical arithmetic, and can favor its draft",
        "alignment. Argmax disagreements and draft acceptance are preserved. This is throughput diagnosis,",
        "not generated-code correctness, SWE-bench, or downstream quality evaluation.",
        "",
        "RAM pressure stayed normal. ASTOP observations include both GPUs, power/frequency/temperature,",
        "per-core CPU activity and exact-process memory/CPU data. GPU utilization can include wait kernels.",
        "Unknown telemetry and ASTOP exit status remain unknown; direct child exits establish completion.",
        "",
        "Frozen inputs and source/binary hashes: manifest.json and phase2-manifest.json.",
        "Full first-phase data: results.json and RESULTS.md. Separated controls: phase2-results.json.",
        "The original deployment binaries/configurations were retained; diagnostic engines ran through",
        "stdin/stdout and opened no network service.",
        "",
        "Raw first phase: " + first["run_dir"],
        "Raw second phase: " + str(raw),
    ]
    (OUT / "CONCLUSIONS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"reference_center": centers, "variants": {
        name: {m: v[m] for m in METRICS} for name, v in b.items()}}))


if __name__ == "__main__":
    main()
