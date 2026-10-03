"""Serial engine-protocol diagnostics, exact exits, and native ASTOP observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
CUSTOM = Path("/home/haneol/work/strata-causal-custom")


def save(p, data):
    p.write_text(json.dumps(data, indent=2) + "\n")


def command(argv, output=None, check=True):
    r = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if output:
        output.write_text(r.stdout)
    if check and r.returncode:
        raise RuntimeError(f"{argv[0]} failed ({r.returncode}): {r.stdout[:700]}")
    return r


def idle(path, strict=True):
    r = command(["astop", "--json"], path)
    data = json.loads(r.stdout)
    cards = [g for g in data["gpus"] if g["vendor_id"] == 4318]
    if len(cards) != 2:
        raise RuntimeError("Expected two NVIDIA cards")
    for g in cards:
        memory = g.get("in_use_system_memory_bytes")
        util = g.get("device_utilization_percent")
        if memory is None or util is None or memory > 512 * 1024**2 or (strict and util > 5):
            raise RuntimeError(f"GPU is occupied or telemetry unavailable: {g['index']}")


def run_one(v, run_dir, inputs, follow):
    run_dir.mkdir()
    idle(run_dir / "before.json")
    repo = ROOT if v["engine"] == "upstream" else CUSTOM
    args = [v.get("exe", str(repo / "build/strata")), *v["args"]]
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="1,0",
               LD_LIBRARY_PATH="/home/haneol/work/strata/.venv/lib/python3.12/site-packages/nvidia/cu13/lib",
               STRATA_DECODE_TIMING="1", STRATA_SPLIT_TIMING="1", **v["env"])
    save(run_dir / "invocation.json", {"argv": args, "env": {k: env[k] for k in
         ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "STRATA_DECODE_TIMING", "STRATA_SPLIT_TIMING", *v["env"])}})
    q = queue.Queue()
    watch = "causal-" + v["name"].lower()
    with (run_dir / "engine.log").open("w") as err, (run_dir / "protocol.jsonl").open("w") as protocol:
        proc = subprocess.Popen(args, cwd=repo, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=err, text=True, bufsize=1)
        save(run_dir / "receipt.json", {"pid": proc.pid, "watch": watch, "started_unix": time.time()})

        def read():
            for line in proc.stdout:
                protocol.write(json.dumps({"unix": time.time(), "line": line.rstrip("\n")}) + "\n")
                protocol.flush()
                q.put(line.rstrip("\n"))
            q.put(None)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        rows = []

        def get(deadline):
            remain = deadline - time.monotonic()
            if remain <= 0:
                raise TimeoutError("Engine protocol deadline")
            line = q.get(timeout=remain)
            if line is None:
                raise RuntimeError(f"Engine protocol closed; exit={proc.poll()}")
            if line.startswith("ERR "):
                raise RuntimeError(line)
            return line

        try:
            command(["astop", "wait", "--pid", str(proc.pid), "--until", "start", "--timeout", "10s", "--json"],
                    run_dir / "observed-start.json")
            command(["astop", "watch", "start", "--name", watch, "--pid", str(proc.pid),
                     "--log", str(run_dir / "engine.log"), "--owner", "codex",
                     "--session", "strata-causal-20261003", "--json"], run_dir / "watch.json")
            deadline = time.monotonic() + 240
            while not get(deadline).startswith("READY "):
                pass
            for index, fixture in enumerate(v["sequence"], 1):
                prompt = inputs[fixture]
                start = time.time()
                proc.stdin.write("GEN 300 " + ",".join(map(str, prompt)) + "\n")
                proc.stdin.flush()
                emitted = []
                deadline = time.monotonic() + 180
                while True:
                    line = get(deadline)
                    if line.startswith("T "):
                        emitted.append(int(line[2:]))
                    elif line.startswith("DONE "):
                        fields = line.split()
                        break
                if not v.get("natural") and emitted != follow:
                    raise RuntimeError(f"{v['name']} request {index}: fixed output mismatch ({len(emitted)} tokens)")
                if (not v.get("natural") and (fields[5] != "length" or int(fields[1]) != 300)) or int(fields[2]) != len(prompt):
                    raise RuntimeError(f"Unexpected DONE: {line}")
                row = {
                    "variant": v["name"], "request": index, "fixture": fixture,
                    "start_unix": start, "end_unix": time.time(), "generated": int(fields[1]),
                    "prompt_tokens": int(fields[2]), "prompt_ms": float(fields[3]),
                    "decode_ms": float(fields[4]), "finish": fields[5],
                    "draft_accepted": int(fields[6]), "draft_offered": int(fields[7]),
                    "reused": int(fields[8]), "extra_done_fields": fields[9:],
                    "output_sha256": hashlib.sha256((" ".join(map(str, emitted)) + "\n").encode()).hexdigest(),
                }
                (run_dir / f"output-{index}.tokens").write_text(" ".join(map(str, emitted)) + "\n")
                row["decode_tok_s"] = row["generated"] * 1000 / row["decode_ms"]
                row["prefill_tok_s"] = (len(prompt) - row["reused"]) * 1000 / row["prompt_ms"]
                rows.append(row)
                save(run_dir / "measurements.json", rows)
                print(f"CAUSAL_REQUEST {v['name']} {fixture} #{index}: prefill={row['prefill_tok_s']:.1f} decode={row['decode_tok_s']:.1f}", flush=True)
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.stdin.close()   # EOF lets the engine's stdin reader join after consuming QUIT.
            rc = proc.wait(timeout=45)
            reader.join(timeout=5)
            save(run_dir / "exit.json", {"exit_code": rc, "requests": len(rows), "fixed_output_all_match": not v.get("natural", False)})
            if rc:
                raise RuntimeError(f"Engine exit {rc}")
            # A device's utilization interval can still contain this process's final work after exit.
            # Require released memory here; enforce both memory and utilization before the next launch.
            idle(run_dir / "after.json", strict=False)
            return rows
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                save(run_dir / "interrupted-exit.json", {"exit_code": proc.returncode})
            reader.join(timeout=5)
            command(["astop", "wait", watch, "--until", "exit", "--timeout", "3s", "--json"],
                    run_dir / "terminal.json", check=False)
            try:
                event = json.loads((run_dir / "terminal.json").read_text())
                event_id = event.get("terminal_event_id") or event.get("event_id")
                if event_id:
                    command(["astop", "event", "ack", event_id, "--consumer", "strata-causal-20261003"],
                            run_dir / "ack.json", check=False)
            except (ValueError, OSError):
                pass
            command(["astop", "watch", "stop", watch], run_dir / "watch-stop.txt", check=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--only", nargs="+")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plan", type=Path, default=ROOT / "docs/causal-diag/plan.json")
    p.add_argument("--manifest", type=Path, default=ROOT / "docs/causal-diag/manifest.json")
    a = p.parse_args()
    a.run_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(a.plan.read_text())
    manifest = json.loads(a.manifest.read_text())
    for item in manifest["files"]:
        with Path(item["path"]).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != item["sha256"]:
                raise RuntimeError(f"Frozen input changed: {item['path']}")
    save(a.run_dir / "plan.json", plan)
    save(a.run_dir / "manifest.json", manifest)
    fixtures = ROOT / "docs/causal-diag/fixtures"
    inputs = {name: list(map(int, (fixtures / f"{name}.tokens").read_text().split()))
              for name in ("code", "edit", "long-code")}
    follow = list(map(int, (fixtures / "follow.tokens").read_text().split()))
    variants = [v for v in plan.get("gates", []) + plan["variants"] if not a.only or v["name"] in a.only]
    results = []
    recorder = subprocess.Popen(["astop", "--record", str(a.run_dir / "telemetry.jsonl"), "--interval", "1s"],
                                stdout=(a.run_dir / "telemetry.log").open("w"), stderr=subprocess.STDOUT)
    try:
        for step, v in enumerate(variants, 1):
            d = a.run_dir / v["name"]
            if a.resume and (d / "exit.json").exists():
                exit_record = json.loads((d / "exit.json").read_text())
                if exit_record["exit_code"] != 0:
                    raise RuntimeError(f"Cannot resume past failed run {v['name']}")
                rows = json.loads((d / "measurements.json").read_text())
            else:
                print(f"CAUSAL_START step={step}/{len(variants)} variant={v['name']}", flush=True)
                rows = run_one(v, d, inputs, follow)
            results += rows
            if v.get("expected_output_sha256"):
                if rows[0]["output_sha256"] != v["expected_output_sha256"]:
                    raise RuntimeError(f"Diagnostic-off output changed: {v['name']}")
            if v["name"] in ("C-gate-patched", "P-gate-patched"):
                original = a.run_dir / v["name"].replace("patched", "original") / "output-1.tokens"
                if (d / "output-1.tokens").read_bytes() != original.read_bytes():
                    raise RuntimeError(f"Diagnostic-off output changed: {v['name']}")
            save(a.run_dir / "measurements.json", results)
            print(f"CAUSAL_STEP {step}/{len(variants)} {v['name']}", flush=True)
        save(a.run_dir / "complete.json", {"exit_code": 0, "variants": len(variants), "requests": len(results)})
        print("CAUSAL_COMPLETE", flush=True)
    except BaseException as e:
        save(a.run_dir / "failure.json", {"error": str(e), "completed_requests": len(results)})
        raise
    finally:
        recorder.terminate()
        recorder.wait(timeout=10)


if __name__ == "__main__":
    main()
