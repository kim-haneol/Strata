"""Functional and bounded throughput checks for the local UD-Q4 expert-tier service."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

FIXTURES = Path('/home/haneol/work/strata/local-runs/compare-20260930T114000Z/fixtures')


def save(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def request(base: str, path: str, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(base + path, data=data, headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--expected-context', type=int, default=131072)
    args = parser.parse_args()
    base = f'http://127.0.0.1:{args.port}'
    end = time.monotonic() + 300
    health = None
    while time.monotonic() < end:
        try:
            health = request(base, '/health', timeout=5)
            if health.get('status') == 'ok':
                break
        except (OSError, URLError):
            pass
        time.sleep(1)
    assert health and health.get('status') == 'ok', 'server did not become healthy'
    assert health['max_context'] == args.expected_context and health['images'] is False, health
    save(args.run_dir / 'health.json', health)
    print('Q4_CHECK step=1 healthy', flush=True)

    common = {'model': health['model'], 'temperature': 0, 'reasoning_effort': 'none'}
    arithmetic = request(base, '/v1/chat/completions', {**common, 'max_tokens': 96, 'messages': [{
        'role': 'user',
        'content': 'Return only JSON with keys multiply (17 times 23), sum (123 plus 456), and sorted (3, 1, 2 ascending).'
    }]})
    save(args.run_dir / 'arithmetic.json', arithmetic)
    text = arithmetic['choices'][0]['message']['content']
    decoded = json.loads(text[text.index('{'):text.rindex('}') + 1])
    assert decoded == {'multiply': 391, 'sum': 579, 'sorted': [1, 2, 3]}, decoded
    print('Q4_CHECK step=2 arithmetic', flush=True)

    rows = []
    for step, name in enumerate(('english-4096-r1', 'retrieve-32768'), 3):
        fixture = json.loads((FIXTURES / f'{name}.json').read_text())
        start = time.monotonic()
        response = request(base, '/v1/chat/completions', {
            **common, 'max_tokens': fixture['max_new'], 'messages': fixture['messages']
        })
        wall = time.monotonic() - start
        save(args.run_dir / f'{name}.json', response)
        output = response['choices'][0]['message']['content']
        if fixture['expected']:
            got = json.loads(output[output.index('{'):output.rindex('}') + 1])
            assert got == fixture['expected'], got
        timing = response['timings']
        row = {'fixture': name, 'prompt_tokens': response['usage']['prompt_tokens'],
               'completion_tokens': response['usage']['completion_tokens'], 'wall_s': wall,
               'prompt_tok_s': timing['prompt_per_second'], 'decode_tok_s': timing['predicted_per_second'],
               'cache_n': timing.get('cache_n'), 'timings': timing}
        rows.append(row)
        save(args.run_dir / 'measurements.json', rows)
        print(f"Q4_CHECK step={step} {name} prefill={row['prompt_tok_s']:.1f} decode={row['decode_tok_s']:.1f}",
              flush=True)
    metrics = request(base, '/metrics')
    save(args.run_dir / 'metrics.json', metrics)
    engine = metrics['engine']
    assert engine['kv'] == 'int8' and engine['context'] == args.expected_context and engine['second_gpu'] == 1, engine
    save(args.run_dir / 'checks.json', {'health': health, 'arithmetic': decoded, 'measurements': rows,
                                        'engine': engine})
    print('Q4_CHECK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
