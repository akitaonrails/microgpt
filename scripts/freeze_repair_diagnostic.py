"""Freeze 96 source-held-out repair cases and a separate training rollout sample."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from prepare_public_horizon import split_for


def tests_for(reference):
    return ('globals().update({k: v for k, v in vars(s).items() if not k.startswith("__")})\n'
            + '\n'.join(reference['testcase']) + '\nassert s._preserved_helper() == "unchanged"\n')


def fixture(reference, before, family, prefix):
    return {'id': prefix + str(reference['seq_id']), 'group': reference['group'],
            'prompt': 'Fix the implementation in solution.py to satisfy this request. Preserve unrelated functions and run check.\n'
                      + reference['prompt'] + '\nRequired signature: ' + reference['signature'],
            'code': before, 'tests': tests_for(reference), 'family': family,
            'transient_check': family == 'recovery'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/structural-repairs-v2'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists(): p.error('Frozen diagnostics must not be overwritten')
    repairs = [json.loads(line) for line in Path('data/public-code-multiseed-v1/repairs.jsonl').open()]
    refs = {r['seq_id']: r for line in Path('data/public-code-multiseed-v1/references.jsonl').open() if (r := json.loads(line))}
    checks = {r['id']: r for line in (args.data / 'checks.jsonl').open() if (r := json.loads(line))['retained']}
    structural = [json.loads(line) for line in (args.data / 'references.jsonl').open()]
    random.Random(20261010).shuffle(repairs)
    random.Random(20261011).shuffle(structural)
    tasks = []; used = set()
    for r in repairs:
        if split_for(r['group']) != 'val' or r['group'] in used: continue
        read = next(m['content'] for m in r['messages'] if m['role'] == 'tool')
        before = '\n'.join(re.findall(r'^\d+: (.*)$', read, flags=re.MULTILINE)) + '\n'
        tasks.append(fixture(refs[r['upstream_id']], before, 'single_mutation', 'single_'))
        used.add(r['group'])
        if len(tasks) == 32: break
    counts = Counter(); train = []
    for r in structural:
        check = checks.get(r['seq_id'])
        if not check: continue
        if split_for(r['group']) == 'train':
            if len(train) < 64:
                train.append(fixture(r, check['before'], 'training_structural', 'train_'))
            continue
        if r['group'] in used: continue
        family = 'structural' if counts['structural'] < 32 else 'recovery'
        if counts[family] == 32: continue
        task = fixture(r, check['before'], family, family + '_')
        task['mutation'] = check['mutation']
        tasks.append(task); used.add(r['group']); counts[family] += 1
    if len(tasks) != 96 or len(train) != 64 or len(used) != 96:
        raise ValueError('Insufficient distinct checked sources for frozen diagnostic')
    assert not used & {t['group'] for t in train}
    args.out.mkdir(parents=True)
    for name, rows in [('evaluation', tasks), ('training-rollouts', train)]:
        (args.out / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report = {'counts': dict(Counter(t['family'] for t in tasks)), 'unique_eval_groups': len(used),
        'train_rollouts': len(train), 'split': 'unchanged SHA256 source-group split',
        'files': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in args.out.iterdir()},
        'note': 'Frozen before new training. Initial fixtures derive from multiseed controls. Runtime injected transient failure is validated separately.'}
    (args.out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
