"""Build multiseed-checked structural repairs and executed recovery lessons.

All public Python runs inside restricted Docker. AST manipulation on the host
never executes source. Recovery lessons are reference-policy demonstrations;
model-produced failures are collected separately.
"""
import argparse
import ast
from collections import Counter
import copy
import hashlib
import inspect
import json
from pathlib import Path
import random

from microgpt.opencode_format import read_output
from prepare_executed_repairs import episode, scoped_edit
from prepare_public_horizon import split_for
from recheck_public_code import sandbox


def structural_mutation(code, seed):
    tree = ast.parse(code)
    rng = random.Random(seed)
    families = ['omitted_guard', 'omitted_loop_body', 'wrong_return', 'empty_implementation']
    rng.shuffle(families)
    for family in families:
        changed = copy.deepcopy(tree)
        candidates = [n for n in ast.walk(changed) if
            (family == 'omitted_guard' and isinstance(n, ast.If)) or
            (family == 'omitted_loop_body' and isinstance(n, (ast.For, ast.While))) or
            (family == 'wrong_return' and isinstance(n, ast.Return) and n.value is not None) or
            (family == 'empty_implementation' and isinstance(n, ast.FunctionDef))]
        if not candidates:
            continue
        node = rng.choice(candidates)
        if family == 'wrong_return':
            node.value = ast.Constant(None)
        elif family == 'empty_implementation':
            node.body = [ast.Raise(exc=ast.Call(func=ast.Name(id='NotImplementedError', ctx=ast.Load()), args=[], keywords=[]), cause=None)]
        else:
            node.body = [ast.Pass()]
        mutated = ast.unparse(ast.fix_missing_locations(changed)) + '\n'
        if mutated != ast.unparse(tree) + '\n':
            return mutated, family
    raise ValueError('Expected a mutable Python function')


def apply_exact(text, old, new):
    if old == new:
        raise ValueError('No changes to apply: oldString and newString are identical.')
    if not old or text.count(old) != 1:
        raise ValueError('Could not find a unique oldString in the file.')
    return text.replace(old, new, 1)


WORKER = inspect.getsource(scoped_edit) + '\n' + inspect.getsource(apply_exact) + r'''
import concurrent.futures, itertools, json, pathlib, subprocess, sys, tempfile
seeds = json.load(open('/settings.json'))['hash_seeds']

def run(root, seed):
    fault = root / '.fault'
    if fault.exists():
        fault.unlink()
        return {'returncode': None, 'output': 'TRANSIENT_ERROR: Test runner temporarily unavailable. Retry check.'}
    try:
        result = subprocess.run([sys.executable, '-s', '-P', '-B', str(root / 'grader.py')], cwd=root,
            env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'PYTHONHASHSEED': str(seed)},
            capture_output=True, text=True, timeout=2)
        return {'returncode': result.returncode, 'output': (result.stdout + result.stderr)[-4000:]}
    except subprocess.TimeoutExpired:
        return {'returncode': None, 'output': 'Grader timed out'}

def check(row):
    result = {'id': row['seq_id'], 'retained': False, 'seeds': []}
    with tempfile.TemporaryDirectory(prefix='structural-repair-') as directory:
        root = pathlib.Path(directory)
        file = root / 'solution.py'
        helper = '\ndef _preserved_helper():\n    return "unchanged"\n'
        before, after = row['broken'] + helper, row['canonical'] + helper
        if before == after:
            result['reason'] = 'Mutation does not change source'
            return result
        tests = '\n'.join(row['testcase'])
        (root / 'grader.py').write_text('import sys\nsys.path.insert(0, ' + repr(str(root)) + ')\nimport solution as s\n'
            + 'globals().update({k: v for k, v in vars(s).items() if not k.startswith("__")})\n'
            + tests + '\nassert s._preserved_helper() == "unchanged"\nprint("PASS")\nraise SystemExit(73)\n')
        old, new = scoped_edit(before, after)
        for seed in seeds:
            file.write_text(before)
            failed = run(root, seed)
            repaired_text = apply_exact(file.read_text(), old, new)
            assert repaired_text == after
            file.write_text(repaired_text)
            repaired = run(root, seed)
            result['seeds'].append({'seed': seed, 'broken_exit': failed['returncode'], 'repaired_exit': repaired['returncode']})
            if failed['returncode'] != 1 or repaired['returncode'] != 73:
                return result
        file.write_text(before)
        errors = {}
        for name, bad_old, bad_new in [('noop', old, old), ('stale', old + '# stale text', new)]:
            try:
                file.write_text(apply_exact(file.read_text(), bad_old, bad_new))
                raise AssertionError('Expected edit rejection')
            except ValueError as error:
                errors[name] = str(error)
            assert file.read_text() == before
        (root / '.fault').write_text('one transient failure')
        transient = run(root, seeds[0])
        assert transient['output'].startswith('TRANSIENT_ERROR') and not (root / '.fault').exists()
        retry = run(root, seeds[0])
        assert retry['returncode'] == 1
        result.update(retained=True, root=str(root), before=before, after=after,
            old_string=old, new_string=new, failed=failed, repaired=repaired,
            mutation=row['mutation'], errors=errors, transient=transient, retry=retry)
    return result

with open('/input.jsonl') as source, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    while True:
        batch = [json.loads(line) for line in itertools.islice(source, 128)]
        if not batch: break
        for result in pool.map(check, batch):
            print(json.dumps(result), flush=True)
'''


def recovery_episode(record, check, tools, kind):
    result = episode(record, check, tools)
    original = result['messages']
    messages = copy.deepcopy(original[:3])
    def action(name, arguments, observation, supervise=True):
        ident = f'recovery_{len(messages)}'
        messages.extend([{'role': 'assistant', 'content': '', 'supervise': supervise,
            'tool_calls': [{'id': ident, 'type': 'function', 'function': {'name': name, 'arguments': arguments}}]},
            {'role': 'tool', 'tool_call_id': ident, 'content': observation}])
    action('read', {'filePath': 'solution.py'}, read_output('solution.py', check['before'], check['root']))
    if kind == 'transient':
        action('check', {}, check['transient']['output'])
    action('check', {}, 'FAIL\n' + check['failed']['output'])
    if kind in ('noop', 'stale'):
        bad = {'filePath': 'solution.py', 'oldString': check['old_string'], 'newString': check['old_string']}
        if kind == 'stale':
            bad.update(oldString=check['old_string'] + '# stale text', newString=check['new_string'])
        action('edit', bad, check['errors'][kind], supervise=False)
        action('read', {'filePath': 'solution.py'}, read_output('solution.py', check['before'], check['root']))
    action('edit', {'filePath': 'solution.py', 'oldString': check['old_string'], 'newString': check['new_string']}, 'Edit applied successfully.')
    action('check', {}, 'PASS\n' + check['repaired']['output'])
    messages.append(copy.deepcopy(original[-1]))
    result.update(messages=messages, recovery=kind,
        construction='Reference-policy recovery. Rejected edit and transient checker fault executed in restricted Docker; incorrect edit target masked. Not a model rollout.')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--train-count', type=int, default=6000)
    p.add_argument('--val-count', type=int, default=192)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    accepted = {r['id'] for line in Path('data/public-code-multiseed-v1/checks.jsonl').open()
                if (r := json.loads(line))['retained']}
    records = [r for line in Path('data/public-code-multiseed-v1/references.jsonl').open()
               if (r := json.loads(line))['seq_id'] in accepted and '_preserved_helper' not in r['code']]
    random.Random(20261010).shuffle(records)
    selected = []; counts = Counter(); groups = set()
    for r in records:
        split = split_for(r['group'])
        if counts[split] >= (args.train_count if split == 'train' else args.val_count) or r['group'] in groups:
            continue
        r['broken'], r['mutation'] = structural_mutation(r['code'], r['seq_id'])
        r['canonical'] = ast.unparse(ast.parse(r['code'])) + '\n'
        selected.append(r); groups.add(r['group']); counts[split] += 1
        if counts['train'] == args.train_count and counts['val'] == args.val_count: break
    source = args.out / 'references.jsonl'
    source.write_text(''.join(json.dumps(r) + '\n' for r in selected)); source.chmod(0o644)
    sandbox(source, args.out / 'checks.jsonl', [1, 2, 3], max(600, len(selected) * 20), worker_source=WORKER)
    checks = {r['id']: r for line in (args.out / 'checks.jsonl').open() if (r := json.loads(line))['retained']}
    tools = json.loads(Path('data/opencode-tools-v2.json').read_text())
    retained = Counter(); families = Counter()
    with (args.out / 'structural.jsonl').open('x') as plain, (args.out / 'recovery.jsonl').open('x') as recovery:
        for i, r in enumerate(selected):
            if r['seq_id'] not in checks: continue
            check = checks[r['seq_id']]
            plain.write(json.dumps(episode(r, check, tools)) + '\n')
            kind = ['noop', 'stale', 'transient'][i % 3]
            recovery.write(json.dumps(recovery_episode(r, check, tools, kind)) + '\n')
            retained[split_for(r['group'])] += 1; families[check['mutation']] += 1
    report = {'selected': dict(counts), 'retained': dict(retained), 'families': dict(families),
        'seeds': [1, 2, 3], 'seed': 20261010, 'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.out.iterdir() if p.is_file()}, 'note': 'Prepared execution-checked data; no model performance claim.'}
    (args.out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
