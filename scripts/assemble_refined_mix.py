"""Assemble a balanced repair subset without changing overall source weights."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from prepare_public_horizon import split_for
from microgpt.opencode_format import cli_user_content


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists(): p.error('Use a fresh file')
    structural = [json.loads(line) for line in Path('data/structural-repairs-v2/structural.jsonl').open()]
    recovery = [json.loads(line) for line in Path('data/structural-repairs-v2/recovery.jsonl').open()]
    old = [json.loads(line) for line in Path('data/public-code-multiseed-v1/repairs.jsonl').open()]
    random.Random(20261010).shuffle(old)
    target = Counter(split_for(r['group']) for r in structural); counts = Counter()
    selected = []
    for row in old:
        split = split_for(row['group'])
        if counts[split] < target[split]:
            selected.append(row); counts[split] += 1
    if counts != target: raise ValueError('Insufficient old repairs for balanced subset')
    corrections = [json.loads(line) for line in Path('data/rollout-corrections-v1/corrections.jsonl').open()]
    if not corrections: raise ValueError('Expected verified corrections from model failures')
    rows = selected + structural + recovery + corrections
    # Keep recorded model histories byte-for-byte; fix only generated prompts.
    for row in selected + structural + recovery:
        for message in row['messages']:
            if message['role'] == 'user':
                text = json.loads(message['content'])
                message['content'] = cli_user_content(text + ' Use read, edit, and check tools.')
    args.out.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report = {'single_mutation': dict(counts), 'structural': dict(target), 'reference_recovery': len(recovery),
        'model_failure_corrections': len(corrections), 'total': len(rows),
        'output_sha256': hashlib.sha256(args.out.read_bytes()).hexdigest(),
        'source': 'executed-public-repair', 'source_weights_unchanged': True,
        'generated_user_format': 'literal surrounding quotes and newlines, matching captured OpenCode 1.18.34 CLI requests',
        'note': 'Roughly equal simple/structural/reference-recovery episode counts within the repair source; bounded model corrections appended. Source-group splits unchanged.'}
    args.out.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
