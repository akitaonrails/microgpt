"""Retokenize the frozen corpus without changing documents or train/val membership.

Tokenizer fitting samples training documents only. Existing manifests retain raw
source revisions, licenses/filter provenance and split rules. No public code runs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from microgpt.tokenizer import Tokenizer, train_bpe
from microgpt.protocol import tokenizer_identity


def documents(path, eos, block=1_000_000):
    tokens = np.memmap(path, dtype='<u4', mode='r')
    start = 0
    for offset in range(0, len(tokens), block):
        for end in np.flatnonzero(tokens[offset:offset + block] == eos) + offset:
            yield tokens[start:int(end)].tolist()
            start = int(end) + 1
    if start != len(tokens):
        raise ValueError(f'Unterminated document: {path}')


def decode(tokenizer, ids):
    # Native whole-document decode preserves UTF-8 across token boundaries.
    return tokenizer.encoding.decode(ids, errors='strict') if tokenizer.encoding else tokenizer.decode(ids)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def convert(job):
    source, target, split, expected = job
    source, target = Path(source), Path(target)
    old, new = Tokenizer.load(source.parent), Tokenizer.load(target.parent)
    path = source / f'{split}.bin'
    if sha(path) != expected['sha256']:
        raise ValueError(f'Source checksum mismatch: {path}')
    target.mkdir(exist_ok=True)
    temporary = target / f'{split}.partial'
    counts = {'tokens': 0, 'documents': 0, 'utf8_bytes': 0}
    with temporary.open('wb') as output:
        for ids in documents(path, old.eos):
            text = decode(old, ids)
            encoded = new.encode(text)
            if new.decode(encoded) != text:
                raise ValueError('Tokenizer text round-trip failed')
            np.asarray(encoded + [new.eos], dtype='<u4').tofile(output)
            counts['documents'] += 1
            counts['tokens'] += len(encoded) + 1
            counts['utf8_bytes'] += len(text.encode())
            if counts['documents'] % 10000 == 0:
                (target / f'{split}.progress.json').write_text(json.dumps(counts) + '\n')
    if counts['documents'] != expected['documents']:
        raise ValueError('Document count changed')
    destination = target / f'{split}.bin'
    temporary.replace(destination)
    counts['sha256'] = sha(destination)
    (target / f'{split}.audit.json').write_text(json.dumps(counts, indent=2) + '\n')
    return source.name, split, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('data/foundation-expanded-v1'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--vocab-size', type=int, default=32768)
    p.add_argument('--sample-bytes', type=int, default=100_000_000)
    p.add_argument('--workers', type=int, default=5)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.source / 'meta.json').read_text())
    old = Tokenizer.load(args.source)
    # Deterministic distributed sampling avoids fitting only the first shard.
    sample_counts = {}
    def sample():
        for item in meta['sources']:
            budget = int(args.sample_bytes * item['weight'])
            total = item['data']['splits']['train']['documents']
            stride = max(1, total // max(1, budget // 4000))
            used = 0
            for index, ids in enumerate(documents(args.source / item['path'] / 'train.bin', old.eos)):
                if index % stride: continue
                text = decode(old, ids)
                yield text
                used += len(text.encode())
                if used >= budget: break
            sample_counts[item['path']] = used
    print('Fitting tokenizer on distributed training-only sample', flush=True)
    tokenizer = train_bpe(sample(), args.vocab_size)
    tokenizer.save(args.out)
    print(json.dumps({'phase': 'tokenizer_ready', 'vocab_size': tokenizer.vocab_size, 'sample_bytes': sample_counts}), flush=True)
    jobs = [(str(args.source / item['path']), str(args.out / item['path']), split, item['data']['splits'][split])
            for item in meta['sources'] for split in ('train', 'val')]
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, split, counts in pool.map(convert, jobs):
            results.setdefault(name, {})[split] = counts
            print(json.dumps({'source': name, 'split': split, **counts}), flush=True)
    output = {'vocab_size': tokenizer.vocab_size, 'eos': tokenizer.eos,
              'tokenizer_fingerprint': tokenizer_identity(tokenizer),
              'parent_manifest_sha256': sha(args.source / 'meta.json'),
              'tokenizer_training_sample_bytes': sample_counts,
              'sources': [{**item, 'data': {'parent': item['data'], 'splits': results[item['path']]}}
                          for item in meta['sources']]}
    (args.out / 'meta.json').write_text(json.dumps(output, indent=2) + '\n')
    print('Corpus complete', flush=True)


if __name__ == '__main__': main()
