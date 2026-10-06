"""Build reference-policy repair lessons with real sandbox checks before and after.

These are deterministic demonstrations, not model rollouts or deep reasoning.
Unlike older simulated repair lessons, each failure and success is executed.
"""
import argparse
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import random
import subprocess
import tempfile
import uuid

from microgpt.opencode_format import read_output, system_prompt
from microgpt.programming_data import GENERAL_PROMPT
from microgpt.protocol import reasoning_instruction
from prepare_verified_code import candidate
from eval_opencode import IMAGE


def mutations(code, seed, limit=4):
    """Change one arithmetic/comparison operation or integer literal at a time."""
    tree = ast.parse(code)
    eligible = []
    for index, node in enumerate(ast.walk(tree)):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            eligible.append((index, 'arithmetic operator'))
        elif isinstance(node, ast.Compare) and isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)):
            eligible.append((index, 'comparison operator'))
        elif isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 100:
            eligible.append((index, 'integer literal'))
    rng = random.Random(seed)
    rng.shuffle(eligible)
    result = []
    replacements = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add,
                    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE,
                    ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
    for index, kind in eligible[:limit]:
        changed = copy.deepcopy(tree)
        node = list(ast.walk(changed))[index]
        if isinstance(node, ast.BinOp):
            node.op = replacements[type(node.op)]()
        elif isinstance(node, ast.Compare):
            node.ops[0] = replacements[type(node.ops[0])]()
        else:
            node.value += 1
        result.append({'code': ast.unparse(ast.fix_missing_locations(changed)) + '\n', 'kind': kind})
    return result


def scoped_edit(before, after):
    """Find a unique whole-line replacement while preserving unchanged surroundings."""
    if before == after:
        raise ValueError('No edit needed')
    old_lines, new_lines = before.splitlines(keepends=True), after.splitlines(keepends=True)
    start = 0
    while start < min(len(old_lines), len(new_lines)) and old_lines[start] == new_lines[start]:
        start += 1
    old_end, new_end = len(old_lines), len(new_lines)
    while old_end > start and new_end > start and old_lines[old_end - 1] == new_lines[new_end - 1]:
        old_end -= 1
        new_end -= 1
    while True:
        old = ''.join(old_lines[start:old_end])
        new = ''.join(new_lines[start:new_end])
        if old and before.count(old) == 1:
            return old, new
        if start:
            start -= 1
        elif old_end < len(old_lines):
            old_end += 1
            new_end += 1
        else:
            return before, after


def random_imports(code):
    """Conservatively identify ordinary imports of random-number APIs."""
    modules = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split('.')[0])
    return modules & {'random', 'secrets'}


# Mount one worker file; the helper tested on the host is copied verbatim into it.
WORKER = inspect.getsource(scoped_edit) + r'''
import json, pathlib, subprocess, sys, tempfile

def check(root):
    try:
        run = subprocess.run([sys.executable, '-I', '-B', 'check.py'], cwd=root,
            capture_output=True, text=True, timeout=2)
        return {'returncode': run.returncode, 'output': (run.stdout + run.stderr)[-6000:], 'timeout': False}
    except subprocess.TimeoutExpired:
        return {'returncode': None, 'output': 'Check exceeded two seconds.', 'timeout': True}

for line in open('/input.jsonl'):
    item = json.loads(line)
    result = {'id': item['seq_id'], 'retained': False}
    with tempfile.TemporaryDirectory(prefix='repair-') as directory:
        root = pathlib.Path(directory)
        solution = root / 'solution.py'
        sentinel = '\ndef _preserved_helper():\n    return "unchanged"\n'
        reference = item['code'] + sentinel
        tests = 'assert _preserved_helper() == "unchanged"\n' + '\n'.join(item['testcase'])
        (root / 'check.py').write_text('exec(compile(open("solution.py").read(), "solution.py", "exec"))\n' + tests)
        solution.write_text(reference)
        positive = check(root)
        if positive['returncode'] == 0:
            for mutation in item['mutations']:
                solution.write_text(mutation['code'] + sentinel)
                before = solution.read_text()
                failed = check(root)
                if failed['timeout'] or failed['returncode'] == 0:
                    continue
                # Execute the same exact-string edit represented in the lesson.
                current = solution.read_text()
                old, new = scoped_edit(before, reference) if item['scoped_edits'] else (before, reference)
                if current.count(old) != 1:
                    raise RuntimeError('Ambiguous edit')
                solution.write_text(current.replace(old, new, 1))
                if solution.read_text() != reference:
                    raise RuntimeError('Scoped edit changed the wrong content')
                repaired = check(root)
                if repaired['returncode'] != 0:
                    continue
                result.update(retained=True, root=str(root), before=before, after=solution.read_text(),
                              old_string=old, new_string=new,
                              failed=failed, repaired=repaired, mutation=mutation['kind'])
                break
    print(json.dumps(result), flush=True)
'''


def episode(record, check, tools):
    root = check['root']
    messages = [{'role': 'system', 'content': reasoning_instruction('none')},
                {'role': 'system', 'content': system_prompt(root, agent_prompt=GENERAL_PROMPT).replace(
                    'Is directory a git repo: yes', 'Is directory a git repo: no')},
                {'role': 'user', 'content': json.dumps('Fix the implementation in solution.py to satisfy this request. '
                    'Preserve unrelated functions and run check.\n' + record['prompt'] + '\nRequired signature: ' + record['signature'])}]
    def action(name, arguments, observation):
        identifier = f'call_{len(messages)}'
        messages.extend([{'role': 'assistant', 'content': '', 'tool_calls': [{'id': identifier,
            'type': 'function', 'function': {'name': name, 'arguments': arguments}}]},
            {'role': 'tool', 'tool_call_id': identifier, 'content': observation}])
    action('read', {'filePath': 'solution.py'}, read_output('solution.py', check['before'], root))
    action('check', {}, 'FAIL\n' + check['failed']['output'])
    action('edit', {'filePath': 'solution.py', 'oldString': check['old_string'], 'newString': check['new_string']},
           'Edit applied successfully.')
    action('check', {}, 'PASS\n' + check['repaired']['output'])
    messages.append({'role': 'assistant', 'content': 'Fixed the implementation. The supplied checks passed, including the unrelated helper check.'})
    return {'group': record['group'], 'source': 'executed-public-repair', 'upstream_id': record['seq_id'],
            'license': 'CC-BY-4.0', 'tools': tools, 'messages': messages, 'mutation': check['mutation'],
            'construction': 'Reference-policy demonstration. File reads, exact-string replacement, and before/after tests executed in restricted Docker. OpenCode-compatible formatting; not an OpenCode or model rollout. Source tests may be incomplete.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--curriculum', type=Path, required=True)
    p.add_argument('--tools', type=Path, default=Path('data/opencode-tools-v2.json'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=1000)
    p.add_argument('--scoped-edits', action='store_true', help='Use the smallest unique changed line span, preserving surrounding functions')
    p.add_argument('--exclude-random-imports', action='store_true', help='Conservatively exclude random/secrets imports in references or tests; not a proof of determinism')
    args = p.parse_args()
    if args.out.exists() or args.count < 1:
        p.error('Use a fresh output and positive count')
    chosen = {}
    for line in args.curriculum.open():
        record = json.loads(line)
        chosen[record['upstream_id']] = record
        if len(chosen) == args.count:
            break
    manifest = json.loads((args.source / 'manifest.json').read_text())
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    selected, found = [], set()
    excluded_random = 0
    wanted = pa.array(list(chosen))
    for shard in manifest['files']:
        path = args.source / shard['path']
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != shard['sha256']:
                raise ValueError('Source hash mismatch')
        parquet = pq.ParquetFile(path)
        indices = pc.indices_nonzero(pc.is_in(parquet.read(columns=['id'])['id'], value_set=wanted))
        if not len(indices):
            continue
        table = parquet.read(columns=['id', 'input', 'output', 'unit_tests', 'average_test_score'])
        for raw in table.take(indices).to_pylist():
            found.add(raw['id'])
            record = candidate(raw)
            if args.exclude_random_imports and random_imports(record['code'] + '\n' + '\n'.join(record['testcase'])):
                excluded_random += 1
                continue
            parent = chosen[raw['id']]
            expected = '```python\n' + record['code'] + '```'
            if parent['messages'][-1]['content'] != expected:
                raise ValueError('Reference differs from the checked curriculum')
            record['group'] = parent['group']
            record['scoped_edits'] = args.scoped_edits
            record['mutations'] = mutations(record['code'], seed=raw['id'])
            if record['mutations'] and '_preserved_helper' not in record['code']:
                selected.append(record)
        del table
        if len(found) == len(chosen):
            break
    if found != set(chosen):
        raise ValueError('Some selected source IDs are missing')
    with tempfile.TemporaryDirectory(prefix='microgpt-repair-') as directory:
        base = Path(directory); base.chmod(0o755)
        source = base / 'input.jsonl'; source.write_text(''.join(json.dumps(r) + '\n' for r in selected)); source.chmod(0o644)
        worker = base / 'worker.py'; worker.write_text(WORKER); worker.chmod(0o644)
        name = 'microgpt-repair-' + uuid.uuid4().hex
        command = ['docker', 'run', '--rm', '--name', name, '--network', 'none', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--pids-limit', '64', '--memory', '512m', '--cpus', '2',
            '--user', '65534:65534', '--tmpfs', '/tmp:rw,noexec,size=16m',
            '--mount', f'type=bind,src={source},dst=/input.jsonl,readonly',
            '--mount', f'type=bind,src={worker},dst=/worker.py,readonly', IMAGE, 'python', '-I', '-B', '/worker.py']
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=max(600, len(selected)*12), check=True)
        finally:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
    checks = [json.loads(line) for line in run.stdout.splitlines()]
    if len(checks) != len(selected) or {r['id'] for r in checks} != {r['seq_id'] for r in selected}:
        raise ValueError('Incomplete repair checks')
    results = {r['id']: r for r in checks}
    tools = json.loads(args.tools.read_text())
    with args.out.open('x') as output:
        for record in selected:
            check = results[record['seq_id']]
            if check['retained']:
                output.write(json.dumps(episode(record, check, tools)) + '\n')
    args.out.with_suffix('.checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    summary = {'dataset': manifest['dataset'], 'revision': manifest['revision'], 'requested': len(chosen),
               'mutable': len(selected), 'retained': sum(r['retained'] for r in checks), 'image': IMAGE,
               'edit_style': 'unique changed line span' if args.scoped_edits else 'whole file',
               'excluded_random_imports': excluded_random,
               'checks_sha256': hashlib.sha256(args.out.with_suffix('.checks.json').read_bytes()).hexdigest(),
               'curriculum_sha256': hashlib.sha256(args.curriculum.read_bytes()).hexdigest(),
               'output_sha256': hashlib.sha256(args.out.read_bytes()).hexdigest(),
               'note': 'Prepared repair data only; no model training or agent capability established.'}
    args.out.with_suffix('.source.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
