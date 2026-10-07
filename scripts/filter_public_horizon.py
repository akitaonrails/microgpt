"""Exclude entire reference-policy sessions containing rejected source answers."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checked', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.checked / 'manifest.json').read_text())
    code = args.checked / 'code.jsonl'
    if sha(code) != manifest['code_sha256']:
        raise ValueError('Changed checked code data')
    with code.open() as source:
        accepted = {json.loads(line)['upstream_id'] for line in source}
    rejected_groups = set()
    groups = set()
    with args.input.open() as source:
        for line in source:
            row = json.loads(line)
            groups.add(row['group'])
            if not set(row['upstream_ids']) <= accepted:
                rejected_groups.add(row['group'])
    counts = {'retained_episodes': 0, 'rejected_episodes': 0}
    split_ids = {'train': set(), 'val': set()}
    with args.out.open('x') as output, args.input.open() as source:
        for line in source:
            row = json.loads(line)
            if row['group'] in rejected_groups:
                counts['rejected_episodes'] += 1
                continue
            output.write(line)
            counts['retained_episodes'] += 1
            split = 'val' if int(hashlib.sha256(row['group'].encode()).hexdigest()[:8], 16) % 20 == 0 else 'train'
            split_ids[split].update(row['upstream_ids'])
    if split_ids['train'] & split_ids['val']:
        raise ValueError('Upstream IDs cross splits')
    report = {**counts, 'retained_sessions': len(groups - rejected_groups),
              'rejected_sessions': sorted(rejected_groups),
              'upstream_ids_by_split': {k: len(v) for k, v in split_ids.items()},
              'input_sha256': sha(args.input), 'output_sha256': sha(args.out),
              'checked_manifest_sha256': sha(args.checked / 'manifest.json'),
              'note': 'Source answers were rechecked across seeds; cumulative compositions retain their earlier single-run execution evidence. No new model rollouts.'}
    args.out.with_suffix('.filter.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
