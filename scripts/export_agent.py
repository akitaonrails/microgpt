"""Export our checkpoint and tokenizer for serving, without optimizer/RNG state."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from microgpt.protocol import tokenizer_identity
from microgpt.tokenizer import Tokenizer


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--canonical-tool-ids', action='store_true', default=None)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Choose a fresh output directory')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.load(args.checkpoint.parent)
    fingerprint = tokenizer_identity(tokenizer)
    if saved.get('data', {}).get('tokenizer_fingerprint') != fingerprint:
        parser.error('Checkpoint/tokenizer mismatch')
    keys = ('model_config', 'model', 'step', 'settings', 'data', 'origin', 'protocol', 'assistant_tokens_seen')
    exported = {key: saved[key] for key in keys if key in saved}
    settings = dict(saved.get('inference_settings', {}))
    settings.setdefault('canonical_tool_ids', saved.get('data', {}).get('canonical_tool_ids', False))
    if args.canonical_tool_ids is not None:
        settings['canonical_tool_ids'] = args.canonical_tool_ids
    exported['inference_settings'] = settings
    args.out.mkdir(parents=True)
    tokenizer.save(args.out, portable=True)
    target = args.out / 'model.pt'
    torch.save(exported, target.with_suffix('.tmp'))
    target.with_suffix('.tmp').replace(target)
    manifest = {'source_checkpoint': str(args.checkpoint), 'source_sha256': checksum(args.checkpoint),
                'model_sha256': checksum(target), 'parameters': sum(t.numel() for t in saved['model'].values()),
                'bytes': target.stat().st_size, 'step': saved['step'], 'origin': saved.get('origin'),
                'tokenizer_fingerprint': fingerprint, 'inference_settings': settings,
                'tokenizer_files': {p.name: checksum(p) for p in args.out.iterdir() if p.name != 'model.pt'},
                'note': 'Inference-only export; resume training from the original full checkpoint.'}
    (args.out / 'deployment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
