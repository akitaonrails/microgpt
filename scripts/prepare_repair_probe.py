"""Freeze source-held-out public repairs and execute positive/negative controls.

Public Python executes only in the existing restricted Docker grader. No model
outputs or benchmark answers enter training; fixtures are diagnostic-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import tempfile

from eval_opencode import docker_command, grade


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=8)
    args = p.parse_args()
    if args.out.exists() or args.count < 1:
        p.error('Fresh output and positive count required')
    heldout = []
    for line in Path('data/public-code-multiseed-v1/repairs.jsonl').open():
        record = json.loads(line)
        if int(hashlib.sha256(str(record['group']).encode()).hexdigest()[:8], 16) % 20 == 0:
            heldout.append(record)
    random.Random(20261007).shuffle(heldout)
    selected = heldout[:args.count]
    wanted = {r['upstream_id'] for r in selected}
    references = {}
    for line in Path('data/public-code-multiseed-v1/references.jsonl').open():
        record = json.loads(line)
        if record['seq_id'] in wanted:
            references[record['seq_id']] = record
    tasks = []; controls = []
    with tempfile.TemporaryDirectory(prefix='microgpt-repair-probe-') as temporary:
        root = Path(temporary); work = root / 'work'; work.mkdir(mode=0o755)
        grader = root / 'grader.py'
        for record in selected:
            reference = references[record['upstream_id']]
            read = next(m['content'] for m in record['messages'] if m['role'] == 'tool')
            before = '\n'.join(re.findall(r'^\d+: (.*)$', read, flags=re.MULTILINE)) + '\n'
            # Preserve ordinary traceback lines rather than embedding every test
            # in one giant exec(...) source line visible to the model.
            tests = 'globals().update({k: v for k, v in vars(s).items() if not k.startswith("__")})\n'
            tests += '\n'.join(reference['testcase']) + '\n'
            tests += 'assert s._preserved_helper() == "unchanged"\n'
            task = {'id': 'public_' + str(record['upstream_id']), 'group': record['group'],
                    'prompt': json.loads(record['messages'][2]['content']),
                    'code': before, 'tests': tests, 'mutation': record['mutation']}
            grader.write_text("import sys\nsys.path.insert(0, '/work')\nimport solution as s\n" + tests)
            grader.chmod(0o644)
            command = docker_command(work, grader, 'microgpt-repair-probe-control')
            (work / 'solution.py').write_text(before)
            negative = grade(command)
            edit = next(c['function']['arguments'] for m in record['messages']
                        for c in m.get('tool_calls', []) if c['function']['name'] == 'edit')
            if before.count(edit['oldString']) != 1:
                raise ValueError('Reconstructed source does not match the reference edit')
            after = before.replace(edit['oldString'], edit['newString'], 1)
            (work / 'solution.py').write_text(after)
            positive = grade(command)
            if negative['passed'] or not positive['passed']:
                raise ValueError(f'Invalid fixture controls: {task["id"]}: {negative}, {positive}')
            tasks.append(task); controls.append({'id': task['id'], 'broken': negative, 'repaired': positive})
    args.out.mkdir(parents=True)
    (args.out / 'tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    (args.out / 'controls.json').write_text(json.dumps({'seed': 20261007, 'split': 'original source-group validation',
        'count': len(tasks), 'controls': controls, 'tasks_sha256': hashlib.sha256((args.out / 'tasks.jsonl').read_bytes()).hexdigest()}, indent=2) + '\n')
    print(f'{len(tasks)} repaired references passed; {len(tasks)} broken fixtures rejected', flush=True)


if __name__ == '__main__':
    main()
