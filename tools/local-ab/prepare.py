"""Prepare a separate upstream Q4 pack and configs without opening a CUDA context."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CUSTOM = Path('/home/haneol/work/strata-custom-5080')
BASE = Path('/home/haneol/work/strata')
PACK = Path('/home/haneol/models/strata/packs/ud-q4-k-xl-upstream-0138')
MTP = Path('/home/haneol/models/strata/mtp/rt')
GGUF = Path('/home/haneol/models/hf/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf')


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main() -> None:
    out = ROOT / 'docs/local-ab'
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, STRATA_GGUF_PY=str(BASE / 'third_party/llama.cpp/gguf-py'))
    subprocess.run([sys.executable, str(ROOT / 'tools/iq_pack.py'), '--gguf', str(GGUF),
                    '--out', str(PACK), '--compat-bf16'], cwd=ROOT, env=env, check=True)
    # Reuse the same profile and MTP bytes as the historical run. No setup refreshes them.
    args = [
        '--pack', str(PACK), '--native', str(GGUF),
        '--expert-profile', str(CUSTOM / 'data/expert-profile.bin'), '--expert-cache', 'auto',
        '--prefill', '16384', '--spec', '4', '--spec-min-p', '0.5',
        '--suffix-draft', '3', '--adapt-every', '1', '--mtp', str(MTP),
        '--max-context', '65536', '--kv', 'int8', '--kv-resident', '0',
        '--vram-reserve-mib', '1024', '--pool-workers', '10',
        '--pcie-mode', 'kernel', '--pcie-frac', '0.2',
    ]
    common = {
        'exe': str(ROOT / 'build/strata'), 'cwd': str(ROOT),
        'tokenizer': str(PACK / 'tokenizer'),
        'model_name': 'qwen3.8-flash-next-ud-q4-k-xl',
        'host': '127.0.0.1', 'port': 22938,
        'lib_dirs': [str(BASE / '.venv/lib/python3.12/site-packages/nvidia/cu13/lib')],
    }
    configs = {
        'layer-split': dict(common, args=args + ['--layer-split', 'auto'], gpu=[1, 0],
                           log=str(ROOT / 'local-runs/layer-split-engine.log')),
        # Omit gpu: a two-element gpu list would make engine_args inject --layer-split.
        'peer-host': dict(common, args=args + ['--peer-device', '1', '--peer-prefill-rows', '0',
                                             '--peer-reserve-mib', '1024'],
                          env={'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '1,0'},
                          log=str(ROOT / 'local-runs/peer-host-engine.log')),
    }
    for name, config in configs.items():
        (out / f'{name}.json').write_text(json.dumps(config, indent=2) + '\n')
    paths = [
        CUSTOM / 'local-runs/q4-final-20260930/checks.json',
        CUSTOM / 'local-runs/q4-final-20260930/short-code-summary.json',
        CUSTOM / 'local-runs/q4-final-20260930/short-code-request.json',
        CUSTOM / 'local-runs/fixed-profile-20260930/follow.tokens',
        CUSTOM / 'data/expert-profile.bin',
        *[MTP / name for name in ('dense.bin', 'dense.txt', 'experts.bin', 'draft_vocab.bin')],
        *[BASE / 'local-runs/compare-20260930T114000Z/fixtures' / f'{name}.json'
          for name in ('english-4096-r1', 'english-32768-r1', 'retrieve-32768')],
        *[PACK / name for name in ('index.txt', 'dense.bin', 'native_experts.txt',
                                  'conversions.json', 'compat-bf16.json')],
        out / 'layer-split.json', out / 'peer-host.json',
    ]
    manifest = {
        'upstream_commit': '99f3dbd0b21d1401b3769e0c0d963913607f380b',
        'upstream_version': '0.1.38', 'status': 'prepared_not_gpu_validated',
        'historical_custom_commit': '5af4d62',
        'historical_custom_binary_sha256': '35e6766ba6264694011a25fd5aeb37cfde42255d61467937199742e23a5e5f69',
        'files': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in paths],
        'gguf_shards': [{'path': str(p), 'bytes': p.stat().st_size}
                        for p in sorted(GGUF.parent.glob('*.gguf'))],
        'limits': ['Historical custom results are not a simultaneous three-mode benchmark.',
                   'The two upstream modes share settings but use different expert placement.',
                   'Speculative lookup policy and numerical kernels differ from the custom fork.',
                   'K-quant MMQ is explicitly enabled in this build; it is off in the stock build.'],
    }
    (out / 'inputs.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('PREPARATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
