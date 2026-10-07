"""Check all frozen repair fixtures through the actual evaluation Docker wrapper."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import uuid
from eval_opencode import docker_command, grade


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks-file', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists(): p.error('Use fresh control output')
    tasks = [json.loads(line) for line in args.tasks_file.open()]
    simple = {r['upstream_id']: r for line in Path('data/public-code-multiseed-v1/repairs.jsonl').open() if (r := json.loads(line))}
    structural = {r['id']: r for line in Path('data/structural-repairs-v2/checks.jsonl').open() if (r := json.loads(line))['retained']}
    def check(task):
        identifier = task['id'].split('_', 1)[1]
        if task['family'] == 'single_mutation':
            record = simple[identifier]
            edit = next(c['function']['arguments'] for m in record['messages']
                        for c in m.get('tool_calls', []) if c['function']['name'] == 'edit')
            assert task['code'].count(edit['oldString']) == 1
            after = task['code'].replace(edit['oldString'], edit['newString'], 1)
        else:
            assert task['code'] == structural[identifier]['before']
            after = structural[identifier]['after']
        with tempfile.TemporaryDirectory(prefix='microgpt-diagnostic-control-') as directory:
            root = Path(directory); work = root / 'work'; work.mkdir(mode=0o755)
            grader = root / 'grader.py'
            grader.write_text("import sys\nsys.path.insert(0, '/work')\nimport solution as s\n" + task['tests'] + '\nprint("PASS")\n')
            grader.chmod(0o644)
            command = docker_command(work, grader, 'microgpt-control-' + uuid.uuid4().hex)
            (work / 'solution.py').write_text(task['code']); negative = grade(command)
            (work / 'solution.py').write_text(after); positive = grade(command)
            return {'id': task['id'], 'broken_rejected': not negative['passed'], 'reference_passed': positive['passed'],
                    'negative_output': negative['output'], 'positive_output': positive['output']}
    with ThreadPoolExecutor(max_workers=4) as pool: results = list(pool.map(check, tasks))
    passed = all(r['broken_rejected'] and r['reference_passed'] for r in results)
    report = {'tasks_sha256': hashlib.sha256(args.tasks_file.read_bytes()).hexdigest(), 'passed': passed,
              'count': len(results), 'results': results}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    if not passed: raise RuntimeError('Frozen diagnostic contains an invalid fixture')
    print(f'{len(results)} broken controls rejected and {len(results)} reference controls passed')


if __name__ == '__main__': main()
