"""Select public coding answers with independently executed positive/negative controls.

Only restricted Docker executes public code. Dataset scores are selection hints,
not proof. No benchmark answers are read. Output is a direct-code SFT curriculum.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import tempfile
import uuid

from microgpt.curate import rows, normalize_reference_tests
from microgpt.programming_data import GENERAL_PROMPT
from microgpt.protocol import reasoning_instruction
from eval_opencode import IMAGE


def parse_tests(value):
    tests = json.loads(value) if isinstance(value, str) else value
    if not isinstance(tests, list) or not tests or not all(isinstance(t, str) for t in tests):
        raise ValueError('Expected a nonempty list of test strings')
    normalized = []
    for test in tests:
        try:
            ast.parse(test.strip())
            normalized.append(test)
        except SyntaxError:
            # Some upstream rows contain one extra JSON-string escaping layer.
            # Decode exactly once; never use eval or blanket backslash replacement.
            normalized.append(json.loads('"' + test + '"'))
    return normalize_reference_tests(normalized)


def candidate(row):
    if row.get('average_test_score') != 1.0:
        raise ValueError('upstream_score')
    blocks = re.findall(r'```(?:python|py)\s*\n(.*?)```', row['output'], re.S)
    if len(blocks) != 1 or not 50 <= len(row['input']) <= 4000 or len(blocks[0]) > 3500:
        raise ValueError('length_or_blocks')
    tree = ast.parse(blocks[0])
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].decorator_list or any(
        not isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom)) for n in tree.body
    ):
        raise ValueError('module_shape')
    fn = functions[0]
    tests = parse_tests(row['unit_tests'])
    if sum(map(len, tests)) > 12000:
        raise ValueError('test_length')
    code = ast.unparse(tree) + '\n'
    # Negative control keeps imports/signature, but cannot implement the task.
    fn.body = [ast.Raise(exc=ast.Call(func=ast.Name(id='NotImplementedError', ctx=ast.Load()), args=[], keywords=[]), cause=None)]
    empty = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    return {'seq_id': row['id'], 'code': code, 'empty': empty, 'testcase': tests,
            'prompt': row['input'], 'signature': f'{fn.name}({ast.unparse(fn.args)})'}


def checked_references(directory, curriculum, count):
    """Recover tests for checked answers without executing downloaded Python."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    chosen = {}
    with curriculum.open() as stream:
        for line in stream:
            record = json.loads(line)
            chosen[record['upstream_id']] = record
            if len(chosen) == count:
                break
    manifest = json.loads((directory / 'manifest.json').read_text())
    wanted, found = pa.array(list(chosen)), set()
    for shard in manifest['files']:
        path = directory / shard['path']
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
            parent = chosen[raw['id']]
            if parent['messages'][-1]['content'] != '```python\n' + record['code'] + '```':
                raise ValueError('Reference differs from the checked curriculum')
            yield {**record, 'group': parent['group']}
        del table
        if len(found) == len(chosen):
            break
    if found != set(chosen):
        raise ValueError('Some selected source IDs are missing')


def split_groups(items):
    """Link identical prompts and function-name-normalized ASTs transitively."""
    parents, code_groups = {}, {}
    def root(group):
        parents.setdefault(group, group)
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group
    identities = {}
    for item in items:
        group = hashlib.sha256(item['prompt'].encode()).hexdigest()
        identities[item['seq_id']] = group
        tree = ast.parse(item['code'])
        name = next(n.name for n in tree.body if isinstance(n, ast.FunctionDef))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                node.name = 'FUNCTION'
            elif isinstance(node, ast.Name) and node.id == name:
                node.id = 'FUNCTION'
        fingerprint = hashlib.sha256(ast.dump(tree).encode()).hexdigest()
        other = code_groups.setdefault(fingerprint, group)
        left, right = sorted((root(group), root(other)))
        parents[right] = left
    return {key: root(group) for key, group in identities.items()}


WORKER = r'''
import concurrent.futures, json, subprocess, sys

def run(code, tests):
    try:
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code + '\n' + '\n'.join(tests)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False

def check(row):
    positive = run(row['code'], row['testcase'])
    negative = run(row['empty'], row['testcase']) if positive else None
    return {'id': row['seq_id'], 'reference_passed': positive, 'empty_passed': negative,
            'retained': positive and negative is False}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for result in pool.map(check, (json.loads(line) for line in open('/input.jsonl'))):
        print(json.dumps(result), flush=True)
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=1000)
    p.add_argument('--per-shard-scan', type=int, default=10000)
    p.add_argument('--seed', type=int, default=20261007)
    args = p.parse_args()
    if args.out.exists() or args.count < 1 or args.per_shard_scan < 1:
        p.error('Use a fresh output and positive limits')
    manifest = json.loads((args.source / 'manifest.json').read_text())
    if manifest['dataset'] != 'nvidia/OpenCodeInstruct':
        raise ValueError('Expected pinned OpenCodeInstruct source')
    rng = random.Random(args.seed)
    selected, counts, seen = [], Counter(), set()
    for shard in manifest['files']:
        path = args.source / shard['path']
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != shard['sha256']:
                raise ValueError('Source checksum mismatch')
        for index, row in enumerate(rows(path)):
            if index >= args.per_shard_scan:
                break
            counts['scanned'] += 1
            try:
                item = candidate(row)
            except (ValueError, SyntaxError, TypeError, KeyError, RecursionError) as error:
                counts['rejected_' + (str(error) if isinstance(error, ValueError) and str(error).isidentifier() else type(error).__name__)] += 1
                continue
            fingerprint = hashlib.sha256(ast.dump(ast.parse(item['code'])).encode()).hexdigest()
            if fingerprint in seen:
                counts['duplicate_ast'] += 1
                continue
            seen.add(fingerprint)
            item['fingerprint'] = fingerprint
            counts['eligible'] += 1
            if len(selected) < args.count:
                selected.append(item)
            else:
                position = rng.randrange(counts['eligible'])
                if position < args.count:
                    selected[position] = item
        print(json.dumps({'shard': shard['path'], 'counts': counts}), flush=True)
    if not selected:
        raise ValueError('No eligible records')
    if len({item['seq_id'] for item in selected}) != len(selected):
        raise ValueError('Duplicate source IDs would make sandbox result association ambiguous')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    checks_path = args.out.with_suffix('.checks.jsonl')
    with tempfile.TemporaryDirectory(prefix='microgpt-verified-') as directory:
        base = Path(directory); base.chmod(0o755)
        source = base / 'input.jsonl'; source.write_text(''.join(json.dumps(r) + '\n' for r in selected)); source.chmod(0o644)
        worker = base / 'worker.py'; worker.write_text(WORKER); worker.chmod(0o644)
        name = 'microgpt-verified-' + uuid.uuid4().hex
        command = ['docker', 'run', '--rm', '--name', name, '--network', 'none', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--pids-limit', '64', '--memory', '1g', '--cpus', '4', '--user', '65534:65534',
            '--tmpfs', '/tmp:rw,noexec,size=16m', '--mount', f'type=bind,src={source},dst=/input.jsonl,readonly',
            '--mount', f'type=bind,src={worker},dst=/worker.py,readonly', IMAGE, 'python', '-I', '-B', '/worker.py']
        try:
            with checks_path.open('x') as log:
                subprocess.run(command, stdout=log, check=True, timeout=max(600, len(selected)*5))
        finally:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
    checks = [json.loads(line) for line in checks_path.open()]
    if len(checks) != len(selected) or {r['id'] for r in checks} != {r['seq_id'] for r in selected}:
        raise ValueError('Incomplete sandbox results')
    retained = {r['id'] for r in checks if r['retained']}
    groups = split_groups(item for item in selected if item['seq_id'] in retained)
    with args.out.open('x') as output:
        for item in selected:
            if item['seq_id'] not in retained:
                continue
            record = {'source': 'verified-opencodeinstruct', 'group': groups[item['seq_id']],
                'upstream_id': item['seq_id'], 'license': 'CC-BY-4.0', 'code_fingerprint': item['fingerprint'],
                'messages': [{'role': 'system', 'content': reasoning_instruction('none')},
                    {'role': 'system', 'content': GENERAL_PROMPT},
                    {'role': 'user', 'content': item['prompt'] + '\nUse the exact function signature ' + item['signature'] + '. Return only Python code in a fenced block.'},
                    {'role': 'assistant', 'content': '```python\n' + item['code'] + '```'}],
                'tools': [], 'verification': 'Reference passed upstream tests in restricted Docker; NotImplementedError control failed. Tests may be incomplete or share source mistakes.'}
            output.write(json.dumps(record) + '\n')
    summary = {'dataset': manifest['dataset'], 'revision': manifest['revision'], 'source_manifest_sha256': hashlib.sha256((args.source/'manifest.json').read_bytes()).hexdigest(),
        'counts': dict(counts), 'selected': len(selected), 'retained': len(retained), 'image': IMAGE, 'seed': args.seed,
        'selection': f'Reservoir among unique exact ASTs in first {args.per_shard_scan} rows per downloaded shard; not a uniform sample of the entire dataset.',
        'split_grouping': 'Transitive union of identical original prompts and function-name-normalized ASTs; not semantic deduplication.',
        'output_sha256': hashlib.sha256(args.out.read_bytes()).hexdigest(), 'checks_sha256': hashlib.sha256(checks_path.read_bytes()).hexdigest()}
    args.out.with_suffix('.source.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
