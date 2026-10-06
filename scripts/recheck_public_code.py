"""Recheck public references and repair controls under multiple Python hash seeds.

Preserves original examples and split groups. Public code executes only inside
restricted Docker. Existing datasets remain immutable for ongoing comparisons.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import uuid

from prepare_verified_code import checked_references
from prepare_executed_repairs import random_imports
from eval_opencode import IMAGE


WORKER = r'''
import concurrent.futures, itertools, json, subprocess, sys
seeds = json.load(open('/settings.json'))['hash_seeds']

def run(code, tests, seed):
    # -I ignores PYTHONHASHSEED. Use safe-path/no-user-site switches and a tiny
    # explicit environment instead, inside the same restricted Docker boundary.
    try:
        result = subprocess.run([sys.executable, '-s', '-P', '-B', '-c',
            code + '\n' + '\n'.join(tests) + '\nraise SystemExit(73)\n'],
            env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'PYTHONHASHSEED': str(seed)},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
        return result.returncode
    except subprocess.TimeoutExpired:
        return None

def check(row):
    result = {'id': row['seq_id'], 'retained': False, 'seeds': []}
    if row.get('excluded_random'):
        result['reason'] = 'random/secrets import in reference or tests'
        return result
    for seed in seeds:
        positive = run(row['code'], row['testcase'], seed)
        negative = run(row['empty'], row['testcase'], seed) if positive == 73 else None
        observed = {'seed': seed, 'reference_exit': positive, 'empty_exit': negative}
        result['seeds'].append(observed)
        if positive != 73 or negative != 1:
            result['reason'] = 'Reference or empty control failed'
            return result
    result['retained'] = True
    if row.get('repair'):
        tests = row['testcase'] + ['assert _preserved_helper() == "unchanged"']
        result['repair_retained'] = True
        result['repair_seeds'] = []
        for seed in seeds:
            before = run(row['repair']['before'], tests, seed)
            after = run(row['repair']['after'], tests, seed)
            result['repair_seeds'].append({'seed': seed, 'before_exit': before, 'after_exit': after})
            if before != 1 or after != 73:
                result['repair_retained'] = False
    return result

with open('/input.jsonl') as source, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    while True:
        batch = [json.loads(line) for line in itertools.islice(source, 256)]
        if not batch:
            break
        for result in pool.map(check, batch):
            print(json.dumps(result), flush=True)
'''


def sandbox(source, destination, seeds, timeout):
    with tempfile.TemporaryDirectory(prefix='microgpt-recheck-') as directory:
        root = Path(directory); root.chmod(0o755)
        worker = root / 'worker.py'; worker.write_text(WORKER); worker.chmod(0o644)
        settings = root / 'settings.json'; settings.write_text(json.dumps({'hash_seeds': seeds})); settings.chmod(0o644)
        name = 'microgpt-recheck-' + uuid.uuid4().hex
        command = ['docker', 'run', '--rm', '--name', name, '--network', 'none', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--pids-limit', '64', '--memory', '1g', '--cpus', '4',
            '--user', '65534:65534', '--tmpfs', '/tmp:rw,noexec,size=16m',
            '--mount', f'type=bind,src={source.resolve()},dst=/input.jsonl,readonly',
            '--mount', f'type=bind,src={worker},dst=/worker.py,readonly',
            '--mount', f'type=bind,src={settings},dst=/settings.json,readonly', IMAGE, 'python', '-I', '-B', '/worker.py']
        try:
            with destination.open('x') as output:
                subprocess.run(command, stdout=output, check=True, timeout=timeout)
        finally:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--curriculum', type=Path, required=True)
    parser.add_argument('--repairs', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=100000)
    parser.add_argument('--hash-seeds', type=int, nargs='+', default=[1, 2, 3])
    args = parser.parse_args()
    if args.count < 1 or any(not 0 <= seed < 2**32 for seed in args.hash_seeds):
        parser.error('Use positive count and valid Python hash seeds')
    args.out.mkdir(parents=True, exist_ok=False)
    repairs = ({r['id']: r for r in json.loads(args.repairs.with_suffix('.checks.json').read_text()) if r['retained']}
               if args.repairs else {})
    ids = set()
    input_path = args.out / 'references.jsonl'
    with input_path.open('x') as output:
        for record in checked_references(args.source, args.curriculum, args.count):
            if record['seq_id'] in ids:
                raise ValueError('Repeated source ID')
            ids.add(record['seq_id'])
            record['excluded_random'] = bool(random_imports(record['code'] + '\n' + '\n'.join(record['testcase'])))
            if record['seq_id'] in repairs:
                repair = repairs[record['seq_id']]
                record['repair'] = {key: repair[key] for key in ('before', 'after')}
            output.write(json.dumps(record) + '\n')
    input_path.chmod(0o644)
    sandbox(input_path, args.out / 'checks.jsonl', args.hash_seeds, max(600, len(ids) * 20))
    checks = [json.loads(line) for line in (args.out / 'checks.jsonl').open()]
    if len(checks) != len(ids) or {r['id'] for r in checks} != ids:
        raise ValueError('Incomplete sandbox checks')
    retained = {r['id'] for r in checks if r['retained']}
    repaired = {r['id'] for r in checks if r.get('repair_retained')}
    for source, name, allowed in [(args.curriculum, 'code.jsonl', retained), (args.repairs, 'repairs.jsonl', repaired)]:
        if source is not None:
            with (args.out / name).open('x') as output, source.open() as incoming:
                for line in incoming:
                    if json.loads(line)['upstream_id'] in allowed:
                        output.write(line)
    report = {'source': str(args.curriculum), 'source_sha256': hashlib.sha256(args.curriculum.read_bytes()).hexdigest(),
              'checked': len(ids), 'references_retained': len(retained), 'repairs_retained': len(repaired),
              'hash_seeds': args.hash_seeds, 'image': IMAGE,
              'checks_sha256': hashlib.sha256((args.out / 'checks.jsonl').read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256((args.out / 'code.jsonl').read_bytes()).hexdigest(),
              'note': 'All references must finish their checks (exit 73) and empty controls fail (exit 1) at every seed. Original text/groups retained. Multiple runs still do not prove complete specifications or determinism.'}
    if args.repairs:
        report['repairs_sha256'] = hashlib.sha256((args.out / 'repairs.jsonl').read_bytes()).hexdigest()
    (args.out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
