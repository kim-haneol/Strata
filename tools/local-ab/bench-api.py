"""Run the saved Q4 API workloads against one manually started upstream mode."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path('/home/haneol/work/strata/local-runs/compare-20260930T114000Z/fixtures')
OLD = Path('/home/haneol/work/strata-custom-5080/local-runs/q4-final-20260930')


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def request(base: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(base + path, data=data, headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=300) as response:
        return json.load(response)


def exact_json(text: str):
    return json.loads(text[text.index('{'):text.rindex('}') + 1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('layer-split', 'peer-host'))
    p.add_argument('run_dir', type=Path)
    p.add_argument('--port', type=int, default=22938)
    a = p.parse_args()
    a.run_dir.mkdir(parents=True, exist_ok=True)
    build = json.loads((ROOT / 'docs/local-ab/build-verification.json').read_text())
    with Path(build['binary']).open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != build['binary_sha256']:
            raise RuntimeError('Engine binary changed after preparation')
    save(a.run_dir / 'build-verification.json', build)
    inputs = json.loads((ROOT / 'docs/local-ab/inputs.json').read_text())
    for item in inputs['files']:
        with Path(item['path']).open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != item['sha256']:
            raise RuntimeError(f"Input changed: {item['path']}")
    base = f'http://127.0.0.1:{a.port}'
    deadline = time.monotonic() + 300
    while True:
        try:
            health = request(base, '/health')
            if health.get('status') == 'ok':
                break
        except (OSError, URLError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError('Server did not become healthy')
        time.sleep(1)
    if health['max_context'] != 65536 or health['images'] is not False:
        raise RuntimeError(f'Unexpected configuration: {health}')
    save(a.run_dir / 'health.json', health)
    common = {'model': health['model'], 'temperature': 0, 'reasoning_effort': 'none'}
    rows = []

    def run(name: str, body: dict, expected=None):
        before = time.monotonic()
        response = request(base, '/v1/chat/completions', dict(common, **body))
        wall = time.monotonic() - before
        save(a.run_dir / f'{name}.json', response)
        if expected is not None and exact_json(response['choices'][0]['message']['content']) != expected:
            raise RuntimeError(f'{name}: exact functional check failed')
        t = response['timings']
        rows.append({'mode': a.mode, 'fixture': name, 'usage': response['usage'],
                     'wall_s': wall, 'timings': t})
        save(a.run_dir / 'measurements.json', rows)
        print(f"AB_CHECK {a.mode} {name} prefill={t['prompt_per_second']:.1f} "
              f"decode={t['predicted_per_second']:.1f} cache={t.get('cache_n')}", flush=True)
        return response

    # Repeat the historical service-check and short-code sequence first.
    run('arithmetic', {'max_tokens': 96, 'messages': [{'role': 'user', 'content':
        'Return only JSON with keys multiply (17 times 23), sum (123 plus 456), and sorted (3, 1, 2 ascending).'}]},
        {'multiply': 391, 'sum': 579, 'sorted': [1, 2, 3]})
    for name in ('english-4096-r1', 'retrieve-32768'):
        fixture = json.loads((FIXTURES / f'{name}.json').read_text())
        run(name, {'max_tokens': fixture['max_new'], 'messages': fixture['messages']},
            fixture['expected'] or None)
    code = json.loads((OLD / 'short-code-request.json').read_text())
    for i in range(1, 4):
        response = run(f'short-code-{i}', code)
        if response['usage']['completion_tokens'] != 300:
            raise RuntimeError('Short code did not produce the requested 300-token sample')
    # Prose at 32K is an additional longer decode workload, not the retrieval's short answer.
    fixture = json.loads((FIXTURES / 'english-32768-r1.json').read_text())
    run('english-32768-r1', {'max_tokens': fixture['max_new'], 'messages': fixture['messages']})
    metrics = request(base, '/metrics')
    save(a.run_dir / 'metrics.json', metrics)
    engine = metrics['engine']
    if engine.get('context') != 65536 or engine.get('kv') != 'int8':
        raise RuntimeError(f'Unexpected engine metrics: {engine}')
    log = (a.run_dir / 'engine.log').read_text(errors='replace')
    if a.mode == 'peer-host':
        if 'peer GPU 1:' not in log or 'layer split: layers' in log:
            raise RuntimeError('Peer tier startup was not proven by the engine log')
    elif 'layer split: layers' not in log:
        raise RuntimeError('Layer split startup was not proven by the engine log')
    save(a.run_dir / 'checks.json', {'health': health, 'engine': engine, 'measurements': rows})
    print('AB_CHECK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
