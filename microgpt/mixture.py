"""Create a tokenizer-checked manifest for separately prepared token streams."""
import argparse
import json
from pathlib import Path
from .tokenizer import Tokenizer
from .protocol import tokenizer_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--source', action='append', required=True, help='relative/path:positive_weight')
    args = parser.parse_args()
    manifest = {'sources': []}
    identity = None
    for spec in args.source:
        path, weight = spec.rsplit(':', 1)
        directory = (args.data / path).resolve()
        if not directory.is_relative_to(args.data.resolve()):
            raise ValueError('Mixture source escapes its directory')
        metadata = json.loads((directory / 'meta.json').read_text())
        tokenizer = Tokenizer.load(directory)
        fingerprint = tokenizer_identity(tokenizer)
        if identity is not None and fingerprint != identity:
            raise ValueError('Mixture tokenizers differ')
        identity = fingerprint
        if not 0 < float(weight) < float('inf'):
            raise ValueError('Weights must be finite and positive')
        manifest['sources'].append({'path': path, 'weight': float(weight), 'data': metadata})
    manifest.update(vocab_size=tokenizer.vocab_size, eos=tokenizer.eos, tokenizer_fingerprint=identity)
    if (args.data / 'meta.json').exists():
        raise ValueError('Mixture manifest already exists')
    tokenizer.save(args.data)
    (args.data / 'meta.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'sources': len(manifest['sources']), 'train_tokens': sum(s['data']['splits']['train']['tokens'] for s in manifest['sources'])}))


if __name__ == '__main__':
    main()
