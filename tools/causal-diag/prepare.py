"""Freeze bounded, serial diagnostic interventions; no GPU work."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CUSTOM = Path("/home/haneol/work/strata-causal-custom")
OLD = Path("/home/haneol/work/strata-custom-5080")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from strata_tokenizer import Tokenizer


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    out = ROOT / "docs/causal-diag"
    fixtures = out / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    old_fixture = OLD / "local-runs/fixed-profile-20260930"
    prompt = list(map(int, (old_fixture / "prompt.tokens").read_text().split()))
    follow = list(map(int, (old_fixture / "follow.tokens").read_text().split()))
    assert len(prompt) == 54 and len(follow) == 300
    pack = Path("/home/haneol/models/strata/packs/ud-q4-k-xl-upstream-0138/tokenizer")
    vocab = json.loads((pack / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tok = Tokenizer(tokens, (pack / "merges.txt").read_text().split("\n"),
                    json.loads((pack / "token_type.json").read_text()))
    filler = tok.encode(("Archive record: inventory, delivery scheduling and storage capacity were reviewed. "
                         "The team retained the original measurements and checked the procedure.\n") * 1500)
    long_prompt = prompt[:3] + filler[:32768 - len(prompt)] + prompt[3:]
    assert len(long_prompt) == 32768
    edit_prompt = prompt[:3] + tok.encode("Existing code:\n") + follow + tok.encode(
        "\nReturn the existing code verbatim, starting with its first line, without commentary.\n") + prompt[-8:]
    inputs = {"code": prompt, "edit": edit_prompt, "long-code": long_prompt}
    for name, ids in dict(inputs, follow=follow).items():
        (fixtures / f"{name}.tokens").write_text(" ".join(map(str, ids)) + "\n")
    fixture_info = {name: {"tokens": len(ids), "sha256": digest(fixtures / f"{name}.tokens")}
                    for name, ids in dict(inputs, follow=follow).items()}

    common = ["--native", "/home/haneol/models/hf/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
              "--expert-profile", str(OLD / "data/expert-profile.bin"),
              "--expert-cache", "2000", "--prefill", "8192",
              "--spec-min-p", "0.5", "--adapt-every", "0",
              "--mtp", "/home/haneol/models/strata/mtp/rt", "--max-context", "65536",
              "--kv", "int8", "--vram-reserve-mib", "1024", "--pool-workers", "10",
              "--pcie-mode", "kernel", "--pcie-frac", "0.2", "--prompt-cache", "0",
              "--spec-follow", str(fixtures / "follow.tokens"), "--serve"]
    cargs = ["--pack", "/home/haneol/models/strata/packs/ud-q4-k-xl-custom"] + common + [
        "--spec", "4", "--spec-lookup", "0", "--diag-window-cap", "8",
        "--main-gpu", "0", "--second-gpu", "1", "--second-gpu-gib", "12",
        "--second-gpu-reserve-mib", "1024", "--second-gpu-prefetch-mb", "8", "--head-split", "0"]
    uargs = ["--pack", "/home/haneol/models/strata/packs/ud-q4-k-xl-upstream-0138"] + common + [
        "--spec", "8", "--mtp-max-t", "4", "--suffix-draft", "0", "--kv-resident", "0"]
    pargs = uargs + ["--peer-device", "1", "--peer-prefill-rows", "0",
                     "--peer-reserve-mib", "1024", "--peer-slots", "4096"]
    largs = uargs + ["--layer-split", "25"]

    def variant(name, engine, args, effect, change=None, extra=None, env=None):
        a = list(args)
        if change:
            for key, value in change.items():
                a[a.index(key) + 1] = str(value)
        a += extra or []
        return {"name": name, "engine": engine, "args": a, "effect": effect,
                "env": env or {}, "sequence": ["code", "code", "code", "edit", "long-code"]}

    variants = [
        variant("C-base", "custom", cargs, "reference"),
        variant("P-base", "upstream", pargs, "reference", env={"STRATA_PEER_HOT": "0.45"}),
        variant("L-base", "upstream", largs, "reference"),
        variant("C-chunk16K", "custom", cargs, "prompt chunk", {"--prefill": 16384}),
        variant("C-single", "custom", cargs, "decode speculation; MTP memory/prompt work retained", extra=["--diag-single-token"]),
        variant("P-single", "upstream", pargs, "decode speculation; MTP memory/prompt work retained",
                extra=["--diag-single-token"], env={"STRATA_PEER_HOT": "0.45"}),
        variant("L-single", "upstream", largs, "decode speculation; MTP memory/prompt work retained", extra=["--diag-single-token"]),
        variant("C-lookup", "custom", cargs, "prompt lookup", {"--spec-lookup": 16}),
        variant("P-lookup", "upstream", pargs, "prompt lookup", {"--suffix-draft": 3}, env={"STRATA_PEER_HOT": "0.45"}),
        variant("L-lookup", "upstream", largs, "prompt lookup", {"--suffix-draft": 3}),
        variant("C-adapt", "custom", cargs, "adaptive cache", {"--adapt-every": 1}),
        variant("P-adapt", "upstream", pargs, "adaptive cache", {"--adapt-every": 1}, env={"STRATA_PEER_HOT": "0.45"}),
        variant("L-adapt", "upstream", largs, "adaptive cache", {"--adapt-every": 1}),
        variant("C-prefetch0", "custom", cargs, "helper prefetch", {"--second-gpu-prefetch-mb": 0}),
        variant("C-prompt-main", "custom", cargs, "helper prompt processing", extra=["--no-prompt-offload"]),
        variant("P-hot0", "upstream", pargs, "peer placement", env={"STRATA_PEER_HOT": "0"}),
    ]
    # Closing references expose drift and reload variance. They are independent engine lifetimes.
    for name in ("L-base", "P-base", "C-base"):
        original = next(v for v in variants if v["name"] == name)
        variants.append(dict(original, name=name + "-repeat"))
    # Profiling is separated from throughput rows because stamps cost launches and synchronization.
    for name in ("C-base", "P-base", "L-base"):
        original = next(v for v in variants if v["name"] == name)
        v = dict(original, name=name.replace("base", "profile"), sequence=["code"], args=list(original["args"]))
        if v["engine"] == "custom":
            v["args"] += ["--window-profile", "--stats"]
        else:
            v["env"] = dict(v["env"], STRATA_VERIFY_PROFILE="1")
        variants.append(v)
    gates = []
    for source, name, old_repo in (("C-base", "C", OLD), ("P-base", "P", Path("/home/haneol/work/strata-upstream-ab"))):
        base = next(v for v in variants if v["name"] == source)
        args = list(base["args"])
        for option in ("--spec-follow", "--diag-window-cap"):
            if option in args:
                index = args.index(option)
                del args[index:index+2]
        if name == "C":
            args[args.index("--second-gpu-prefetch-mb") + 1] = "0"
        for kind in ("original", "patched"):
            repo = old_repo if kind == "original" else (CUSTOM if name == "C" else ROOT)
            gates.append(dict(base, name=f"{name}-gate-{kind}", args=args, sequence=["code"],
                              natural=True, exe=str(repo / "build/strata")))
    plan = {
        "status": "frozen_before_execution", "context": 65536, "model": "UD-Q4_K_XL",
        "physical_gpu_order": [1, 0], "main_cache_budget_max_blob_slots": 2000,
        "custom_helper_budget_gib": 12, "peer_slot_cap": 4096, "verify_capacity": 8,
        "mtp_window_cap_including_current_token": 4, "fixtures": fixture_info,
        "variants": variants, "gates": gates, "limits": [
            "Teacher-forced continuation is a mechanical throughput diagnostic, not generated-code quality.",
            "Matching tokens does not guarantee identical routed experts or draft acceptance across numerical implementations.",
            "Single-token mode retains the MTP allocation and prompt processing; it isolates decode speculation, not startup or memory savings.",
            "Caches remain adaptive only in the explicit adapt interventions.",
            "Cache placements and architecture differ across C/P/L; their cross-mode residual is a combined implementation effect.",
            "Intervention effects are conditional, not additive percentages of the historical speed gap.",
            "Three code requests share one engine; only reference reloads are independent repetitions.",
            "The primary budget uses max-blob slot units, not a guaranteed count of mixed-size native experts.",
        ],
    }
    save(out / "plan.json", plan)
    files = [fixtures / f"{name}.tokens" for name in (*inputs, "follow")]
    files += [OLD / "data/expert-profile.bin"]
    files += [Path("/home/haneol/models/strata/mtp/rt") / n
              for n in ("dense.bin", "dense.txt", "experts.bin", "draft_vocab.bin")]
    for engine, repo in (("upstream", ROOT), ("custom", CUSTOM)):
        files += [repo / "build/strata"]
    files += [OLD / "build/strata", Path("/home/haneol/work/strata-upstream-ab/build/strata")]
    source_commits = {name: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
                      for name, repo in (("upstream", ROOT), ("custom", CUSTOM))}
    save(out / "manifest.json", {"source_commits": source_commits, "files": [
        {"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in files
    ], "fixtures": fixture_info})
    print(f"CAUSAL_PREPARED variants={len(variants)} requests={sum(len(v['sequence']) for v in variants)}")


if __name__ == "__main__":
    main()
