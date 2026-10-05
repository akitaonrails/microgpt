"""Build repair and task-switch lessons from previously sandbox-checked references.

Tool observations are simulated, not model rollouts. No imported code executes.
Related tasks are combined only within their existing train/validation partition.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import random

from microgpt.opencode_format import read_output, system_prompt
from microgpt.programming_data import GENERAL_PROMPT
from microgpt.protocol import reasoning_instruction


def partition(group):
    return int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 20 == 0


def reference(record):
    edit = next(m['tool_calls'][0]['function']['arguments'] for m in record['messages']
                if m.get('tool_calls') and m['tool_calls'][0]['function']['name'] == 'edit')
    code = edit['newString']
    tree = ast.parse(code)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    broken = copy.deepcopy(tree)
    target = next(n for n in broken.body if isinstance(n, ast.FunctionDef))
    target.body = [ast.Raise(exc=ast.Call(func=ast.Name(id='NotImplementedError', ctx=ast.Load()), args=[], keywords=[]), cause=None)]
    ast.fix_missing_locations(broken)
    request = json.loads(next(m['content'] for m in record['messages'] if m['role'] == 'user'))
    request = request.removesuffix(' Implement this in solution.py and run check.')
    request += f' Use the function signature {function.name}({ast.unparse(function.args)}).'
    return request, code, ast.unparse(broken) + '\n'


def episode(records, seed, repair):
    rng = random.Random(seed)
    root = f'/tmp/programming-{seed}/work'
    messages = [{'role': 'system', 'content': reasoning_instruction('none')},
                {'role': 'system', 'content': system_prompt(root, agent_prompt=GENERAL_PROMPT)}]
    def action(name, arguments, result):
        call = f'call_{len(messages)}'
        messages.extend([{'role': 'assistant', 'content': '', 'tool_calls': [{'id': call, 'type': 'function',
            'function': {'name': name, 'arguments': arguments}}]}, {'role': 'tool', 'tool_call_id': call, 'content': result}])
    for turn, record in enumerate(records):
        request, code, broken = reference(record)
        path = rng.choice(['helpers', 'utils', 'solution', 'module']) + f'_{turn}.py'
        before = broken if repair else '# Python utilities\n'
        instruction = ('Fix the incomplete implementation. ' if repair else 'Implement this request. ')
        messages.append({'role': 'user', 'content': json.dumps(f'{instruction}{request} Work in {path}, preserve other files, and run check.')})
        action('read', {'filePath': path}, read_output(path, before, root))
        if repair:
            action('check', {}, 'FAIL\nNotImplementedError\n')
        action('edit', {'filePath': path, 'oldString': before, 'newString': code}, 'Edit applied successfully.')
        action('check', {}, 'PASS\nPASS\n')
        messages.append({'role': 'assistant', 'content': f'Updated {path}. The check passed.'})
    return {'group': records[0]['group'], 'source': 'programming-recovery' if repair else 'programming-switch',
            'tools': records[0]['tools'], 'messages': messages, 'license': 'MIT',
            'upstream_groups': [r['group'] for r in records],
            'construction': 'Simulated tool observations using previously sandbox-checked OpenCoder references; not independently executed multi-file sessions.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('data/educational-tools-v1.jsonl'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    pools = {False: [], True: []}
    for line in args.input.open():
        r = json.loads(line)
        try: reference(r)
        except (SyntaxError, StopIteration, ValueError): continue
        pools[partition(r['group'])].append(r)
    count = 0
    with args.out.open('x') as out:
        for pool in pools.values():
            for i, record in enumerate(pool):
                for repair, length in ((True, 1), (True, 3), (False, 6)):
                    records = [pool[(i + j * 37) % len(pool)] for j in range(length)]
                    item = episode(records, i + count * 10000, repair)
                    assert all(partition(r['group']) == partition(item['group']) for r in records)
                    out.write(json.dumps(item) + '\n'); count += 1
    args.out.with_suffix('.source.json').write_text(json.dumps({'episodes': count,
        'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(args.out.read_bytes()).hexdigest(),
        'partition_policy': 'Every constituent retains the original hashed train/validation partition.'}, indent=2) + '\n')
    print(f'Wrote {count} repair/task-switch episodes', flush=True)


if __name__ == '__main__':
    main()
