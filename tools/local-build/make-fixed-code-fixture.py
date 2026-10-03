"""Make token files for a fixed-text command-line profile from the saved API response."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACK = Path('/home/haneol/models/strata/packs/ud-q4-k-xl-custom/tokenizer')
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from serve.frontend import ChatTemplate
from strata_tokenizer import Tokenizer

PROMPT = ('Write a complete Python function that parses a CSV string, groups rows by the first column, sums the '
          'integer second column, and returns a deterministically sorted dictionary. Include type hints, validation, '
          'and concise examples.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('response', type=Path)
    args = parser.parse_args()
    vocab = json.loads((PACK / 'vocab.json').read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tok = Tokenizer(tokens, (PACK / 'merges.txt').read_text().split('\n'),
                    json.loads((PACK / 'token_type.json').read_text()))
    rendered = ChatTemplate(PACK / 'chat_template.jinja').render(
        [{'role': 'user', 'content': PROMPT}], enable_thinking=False)
    output = json.loads(args.response.read_text())['choices'][0]['message']['content']
    prompt_ids = tok.encode(rendered, parse_special=True)
    follow_ids = tok.encode(output, parse_special=True)
    (args.run_dir / 'prompt.tokens').write_text(' '.join(map(str, prompt_ids)) + '\n')
    (args.run_dir / 'follow.tokens').write_text(' '.join(map(str, follow_ids)) + '\n')
    meta = {'prompt_tokens': len(prompt_ids), 'follow_tokens': len(follow_ids),
            'prompt_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
            'output_sha256': hashlib.sha256(output.encode()).hexdigest()}
    (args.run_dir / 'fixed-fixture.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
