"""Freeze seen-training and validation examples for an in-distribution diagnostic.

This is explicitly not an independent programming benchmark. Reconstructs the
first unweighted training samples for seed 42; public code executes only later
inside the evaluation sandbox. Never add diagnostic answers to new training data.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random

from microgpt.curate import normalize_reference_tests, rows
from microgpt.episodes import Episodes
from microgpt.protocol import digest
from prepare_interface_curriculum import rename


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--curriculum', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a fresh output')
    manifest = json.loads((args.data / 'manifest.json').read_text())
    episodes = {split: Episodes(args.data / f'{split}.jsonl', manifest['splits'][split]['sha256'])
                for split in ('train', 'val')}
    buckets = episodes['train'].buckets(8192)
    if len(buckets) != 1:
        raise ValueError('This diagnostic expects the single-source unweighted curriculum')
    training = [[index for bucket in buckets.values() for index in bucket]]
    if len(training[0]) < 20 or len(episodes['val']) < 20:
        raise ValueError('Need at least 20 examples in each split')
    rng = random.Random(42)
    selected = {}
    while len(selected) < 20:
        index = rng.choice(rng.choice(training))
        selected[episodes['train'][index]['episode']] = 'seen_train'
    for index in random.Random(20261006).sample(range(len(episodes['val'])), 20):
        selected[episodes['val'][index]['episode']] = 'validation'
    records = {}
    for line in args.curriculum.open():
        record = json.loads(line)
        identity = digest({'messages': record['messages'], 'tools': record.get('tools', [])})
        if identity in selected:
            records[identity] = record
    if len(records) != len(selected):
        raise ValueError('Curriculum does not match prepared episode identities')
    needed = {r['upstream_id'] for r in records.values()}
    upstream = {r['seq_id']: r for r in rows(args.source) if r['seq_id'] in needed}
    with args.out.open('x') as output:
        for number, (identity, split) in enumerate(selected.items()):
            record = records[identity]
            original = upstream[record['upstream_id']]
            code = record['messages'][-1]['content'].removeprefix('```python\n').removesuffix('```')
            old = next(n.name for n in ast.parse(original['code']).body if isinstance(n, ast.FunctionDef))
            new = next(n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef))
            tests = rename(ast.parse('\n'.join(normalize_reference_tests(original['testcase']))), old, new)
            task = {'id': f'{split}_{number}', 'family': split, 'upstream_id': record['upstream_id'],
                    'prompt': next(m['content'] for m in record['messages'] if m['role'] == 'user'),
                    'reference': code, 'tests': 'globals().update(vars(s))\n' + tests + '\n'}
            output.write(json.dumps(task) + '\n')
    with args.out.open('rb') as source:
        checksum = hashlib.file_digest(source, 'sha256').hexdigest()
    args.out.with_suffix('.source.json').write_text(json.dumps({
        'data_manifest': manifest, 'tasks_sha256': checksum, 'tasks': len(selected),
        'note': '20 early seed-42 training samples plus 20 validation samples; not held-out capability evidence. SFT validation may have earlier foundation exposure.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
