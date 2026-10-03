"""Summarize the completed upstream pair and frozen custom reference without GPU work."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
CUSTOM = Path('/home/haneol/work/strata-custom-5080/local-runs/q4-final-20260930')


def read(path: Path):
    return json.loads(path.read_text())


def telemetry(path: Path, metrics: dict):
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    decode = []
    for sample in samples:
        at = datetime.fromisoformat(sample['timestamp_utc'].replace('Z', '+00:00')).timestamp()
        for req in metrics['requests']:
            start = req['time'] + req['prompt_ms'] / 1000
            end = start + req['decode_ms'] / 1000
            if start <= at <= end:
                decode.append(sample)
                break
    gpu = []
    for index in (0, 1):
        all_rows = [g for s in samples for g in s['gpus'] if g['index'] == index and g['vendor_id'] == 4318]
        dec_rows = [g for s in decode for g in s['gpus'] if g['index'] == index and g['vendor_id'] == 4318]
        util = [g['device_utilization_percent'] for g in dec_rows if g['device_utilization_percent'] is not None]
        gpu.append({
            'physical_gpu': index,
            'max_vram_gib': max(g['in_use_system_memory_bytes'] for g in all_rows) / 2**30,
            'max_util_percent': max(g['device_utilization_percent'] for g in all_rows
                                    if g['device_utilization_percent'] is not None),
            'max_power_w': max(g['power_watts'] for g in all_rows if g['power_watts'] is not None),
            'max_temperature_c': max(g['temperature_c'] for g in all_rows if g['temperature_c'] is not None),
            'decode_mean_util_percent': statistics.mean(util) if util else None,
            'decode_max_util_percent': max(util) if util else None,
        })
    return {'samples': len(samples), 'approximate_decode_samples': len(decode), 'gpus': gpu,
            'max_used_ram_gib': max(s['memory']['used_bytes'] for s in samples) / 2**30,
            'memory_pressure': sorted({s['memory']['pressure'] for s in samples}),
            'swap_start_gib': samples[0]['memory']['swap_used_bytes'] / 2**30,
            'swap_end_gib': samples[-1]['memory']['swap_used_bytes'] / 2**30,
            'phase_note': 'Decode windows approximate server start plus reported prompt/decode time; 1-second samples.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_root', type=Path)
    ap.add_argument('--output', type=Path, default=ROOT / 'docs/local-ab/results-20261003.json')
    a = ap.parse_args()
    summary = {'date': '2026-10-03', 'upstream_commit': '99f3dbd0b21d1401b3769e0c0d963913607f380b',
               'upstream_version': '0.1.38', 'run_root': str(a.run_root.resolve()),
               'p2p': read(a.run_root / 'p2p.json'), 'modes': {}, 'custom_rerun': False}
    for mode in ('layer-split', 'peer-host'):
        run = a.run_root / mode
        checks = read(run / 'checks.json')
        if read(run / 'bench-result.json')['exit_code'] != 0:
            raise RuntimeError(f'{mode}: benchmark failed')
        if len(checks['measurements']) != 7:
            raise RuntimeError(f'{mode}: incomplete measurements')
        rows = {row['fixture']: row for row in checks['measurements']}
        rates = [rows[f'short-code-{i}']['timings']['predicted_per_second'] for i in (1, 2, 3)]
        if any(rows[f'short-code-{i}']['usage']['completion_tokens'] != 300 for i in (1, 2, 3)):
            raise RuntimeError('Code workload length changed')
        post = read(run / 'post-stop-astop.json')
        nvidia = [g for g in post['gpus'] if g['vendor_id'] == 4318]
        released = all(g['device_utilization_percent'] == 0 and
                       g['in_use_system_memory_bytes'] < 16 * 2**20 for g in nvidia)
        if len(nvidia) != 2 or not released:
            raise RuntimeError('GPU release was not proven')
        log = (run / 'engine.log').read_text()
        peer = re.search(r'peer GPU 1: (\d+) experts', log)
        split = re.search(r'layer split auto: K=(\d+)', log)
        chunk = re.search(r'prompt chunk 16384 -> (\d+) tokens', log)
        metrics = read(run / 'metrics.json')
        tel = telemetry(run / 'telemetry.jsonl', metrics)
        summary['modes'][mode] = {
            'passed': True, 'request_count': 7, 'engine': checks['engine'],
            'actual_prompt_chunk': int(chunk[1]) if chunk else 16384,
            'split_first_stage_layers': int(split[1]) if split else None,
            'peer_expert_slots': int(peer[1]) if peer else 0,
            'prefill_4k_tok_s': rows['english-4096-r1']['timings']['prompt_per_second'],
            'decode_4k_tok_s': rows['english-4096-r1']['timings']['predicted_per_second'],
            'prefill_32k_retrieval_tok_s': rows['retrieve-32768']['timings']['prompt_per_second'],
            'decode_32k_prose_tok_s': rows['english-32768-r1']['timings']['predicted_per_second'],
            'code_decode_tok_s': rates, 'code_decode_median_tok_s': statistics.median(rates),
            'measurements': checks['measurements'], 'telemetry': tel,
            'server_stop_record': (run / 'server-systemd-stop.txt').read_text(),
            'gpu_memory_released': released,
        }
    old = read(CUSTOM / 'checks.json')
    oc = read(CUSTOM / 'short-code-summary.json')
    om = {r['fixture']: r for r in old['measurements']}
    summary['historical_custom'] = {
        'source': str(CUSTOM), 'date': '2026-09-30', 'context': old['engine']['context'],
        'prefill_4k_tok_s': om['english-4096-r1']['prompt_tok_s'],
        'prefill_32k_retrieval_tok_s': om['retrieve-32768']['prompt_tok_s'],
        'code_decode_tok_s': [r['decode_tok_s'] for r in oc['runs']],
        'code_decode_median_tok_s': oc['decode_tok_s']['median'],
    }
    proof = []
    for path in sorted(p for p in a.run_root.rglob('*') if p.is_file() and
                       p.suffix in ('.json', '.jsonl', '.txt', '.log')):
        with path.open('rb') as f:
            proof.append({'path': str(path.resolve()), 'sha256': hashlib.file_digest(f, 'sha256').hexdigest()})
    summary['evidence'] = proof
    summary['limits'] = [
        'Custom is a historical reference, not a contemporary rerun.',
        'Both upstream modes chose 8192 rather than the requested 16384 prompt chunk.',
        'Code outputs have the same requested length, not necessarily the same generated text.',
        'The upstream speculative implementation and lookup differ from the custom fork.',
        'Upstream warm code reused 47 prompt tokens; the custom second run reused 53.',
        'One sequence per upstream mode; no statistical significance is claimed.',
        'A launcher PATH failure preceded all measured requests and did not restart the engine.',
        'Layer-split server stop returned uv status 130 after the requested stop, not during a benchmark.',
    ]
    a.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({name: {k: v for k, v in row.items() if k in
                            ('prefill_4k_tok_s','prefill_32k_retrieval_tok_s','code_decode_tok_s',
                             'decode_32k_prose_tok_s','gpu_memory_released')}
                      for name, row in summary['modes'].items()}, indent=2))


if __name__ == '__main__':
    main()
