"""Recheck saved direct-code outputs after grader changes, without model inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import uuid

from eval_opencode import docker_command, grade, IMAGE, CHECK_SUCCESS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, nargs='+', required=True)
    p.add_argument('--tasks-file', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error('Use a fresh output file')
    tasks = {r['id']: r for r in map(json.loads, args.tasks_file.read_text().splitlines())}
    report = {'image': IMAGE, 'success_exit': CHECK_SUCCESS,
              'tasks_sha256': hashlib.sha256(args.tasks_file.read_bytes()).hexdigest(), 'runs': []}
    for path in args.results:
        original = json.loads(path.read_text())
        if original['mode'] != 'direct':
            raise ValueError('Only direct-code outputs can be regraded by this tool')
        def check(row):
            code = path.parent / (row['id'] + '.py')
            with tempfile.TemporaryDirectory(prefix='microgpt-regrade-') as directory:
                base = Path(directory)
                base.chmod(0o755)
                work = base / 'work'
                work.mkdir(mode=0o755)
                (work / 'solution.py').write_bytes(code.read_bytes())
                grader = base / 'grader.py'
                grader.write_text("import sys\nsys.path.insert(0, '/work')\nimport solution as s\n"
                                  + tasks[row['id']]['tests'] + "\nprint('PASS')\n")
                outcome = grade(docker_command(work, grader, 'microgpt-regrade-' + uuid.uuid4().hex))
            return {'id': row['id'], 'old_passed': row['passed'],
                    'passed': outcome['passed'] and row['completed'],
                    'code_sha256': hashlib.sha256(code.read_bytes()).hexdigest()}
        with ThreadPoolExecutor(max_workers=4) as pool:
            checked = list(pool.map(check, original['tasks']))
        record = {'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'old_passed': sum(r['old_passed'] for r in checked),
                  'passed': sum(r['passed'] for r in checked), 'total': len(checked),
                  'changed': [r for r in checked if r['old_passed'] != r['passed']], 'tasks': checked}
        report['runs'].append(record)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in record.items() if k != 'tasks'}), flush=True)


if __name__ == '__main__':
    main()
