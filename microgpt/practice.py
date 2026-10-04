"""Original, deterministic tool curricula. No downloaded code is executed.

These are simulated multi-file exercises, not real repository trajectories.
All expected edits are checked against exact source targets. Evaluation uses
separate tasks and real OpenCode; training demonstrations never establish skill.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from .download_sft import function, STRING

TOOLS = [
    function('read', {'filePath': STRING}, ['filePath']),
    function('edit', {'filePath': STRING, 'oldString': STRING, 'newString': STRING}, ['filePath', 'oldString', 'newString']),
    function('check', {}, []),
    function('todowrite', {'todos': {'type': 'array', 'items': {'type': 'object', 'properties': {
        'content': STRING, 'status': {'type': 'string', 'enum': ['pending', 'in_progress', 'completed', 'cancelled']},
        'priority': {'type': 'string', 'enum': ['high', 'medium', 'low']}}, 'required': ['content', 'status', 'priority'],
        'additionalProperties': False}}}, ['todos']),
]


def episode(seed, files_count, padding=0, handoff=False):
    rng = random.Random(seed)
    base = f'/workspace/project_{rng.randrange(100000, 999999)}'
    files, targets, descriptions = {}, {}, []
    paths = []
    for index in range(files_count):
        name = f'convert_{rng.randrange(1000, 9999)}_{index}'
        path = f'{base}/{name}.py'
        factor, offset = rng.randint(2, 20), rng.randint(1, 50)
        comment = ''.join(f'# Reference {n}: keep this line unchanged.\n' for n in range(padding))
        files[path] = comment + f'def {name}(value):\n    return value * {factor + 1} + {offset}\n'
        targets[path] = comment + f'def {name}(value):\n    return value * {factor} + {offset}\n'
        descriptions.append(f'{path}: {name}(value) must return value * {factor} + {offset}. Preserve comments.')
        paths.append(path)
    messages = [{'role': 'system', 'content': 'Read files before editing. Make a plan for multiple files. Use exact source strings without displayed line numbers. Recover from tool errors, run check, and report its result.'},
                {'role': 'user', 'content': 'Fix these conversions. Keep every earlier requirement.\n' + '\n'.join(descriptions)}]
    calls = 0
    failure_count = 0
    def act(name, arguments, content=''):
        nonlocal calls, failure_count
        calls += 1
        call_id = f'call_{seed}_{calls}'
        messages.append({'role': 'assistant', 'content': content, 'tool_calls': [
            {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': arguments}}]})
        if name == 'read':
            path = arguments['filePath']
            numbered = '\n'.join(f'{i}: {line}' for i, line in enumerate(files[path].splitlines(), 1))
            result = f'<path>{path}</path>\n<type>file</type>\n<content>\n{numbered}\n</content>'
        elif name == 'edit':
            path, old, new = (arguments[k] for k in ('filePath', 'oldString', 'newString'))
            if not old or files[path].count(old) != 1:
                failure_count += 1
                result = 'Error: oldString was not found exactly once. Read the current file and retry with exact text.'
            else:
                files[path] = files[path].replace(old, new, 1)
                result = 'Edit applied successfully.'
        elif name == 'check':
            failures = [p for p in paths if files[p] != targets[p]]
            result = ('FAIL: conversion requirements still unmet in ' + ', '.join(failures)) if failures else 'PASS: all expected source transformations match.'
        elif name == 'todowrite':
            result = json.dumps(arguments['todos'])
        else:
            raise ValueError(name)
        messages.append({'role': 'tool', 'tool_call_id': call_id, 'content': result})
        return result
    def todos(done):
        return {'todos': [{'content': f'Fix {p}', 'status': 'completed' if i < done else 'pending', 'priority': 'medium'} for i, p in enumerate(paths)]}
    if handoff:
        state_path = base + '/TASK_STATE.md'
        # The harness writes factual task state before starting a fresh session.
        files[state_path] = targets[state_path] = '# Task requirements\n' + '\n'.join(descriptions)
        messages[1]['content'] = f'Resume this coding task after a session handoff. Read {state_path} for the requirements, inspect the files, finish the work, and check it.'
        act('read', {'filePath': state_path}, 'I will recover the task requirements before making changes.')
    if files_count > 1:
        act('todowrite', todos(0), 'I will inspect each conversion, make the requested changes, and check the complete result.')
    for index, path in enumerate(paths):
        conflict = seed % 4 == 0 and index == files_count // 2
        if conflict:
            files[path] = '# Original revision.\n' + files[path]
            targets[path] = '# Original revision.\n' + targets[path]
        act('read', {'filePath': path})
        old = files[path]
        if conflict:
            # Simulate another writer changing a comment between read and edit.
            files[path] = old.replace('# Original revision.', '# Updated by another worker.', 1)
            targets[path] = targets[path].replace('# Original revision.', '# Updated by another worker.', 1)
            act('edit', {'filePath': path, 'oldString': old, 'newString': targets[path]})
            act('read', {'filePath': path}, 'The source changed. I will read it again and preserve the new comment.')
            old = files[path]
        act('edit', {'filePath': path, 'oldString': old, 'newString': targets[path]})
        if files_count > 1 and (index + 1) % 4 == 0:
            act('todowrite', todos(index + 1), 'I will keep the completed changes and continue with the remaining files.')
        if index + 1 < files_count and index % 3 == 0:
            act('check', {}, 'Checking progress before continuing.')
    assert act('check', {}, 'Checking all requested conversions.').startswith('PASS')
    assert files == targets
    if files_count > 1:
        act('todowrite', todos(files_count))
    messages.append({'role': 'assistant', 'content': f'Updated {files_count} conversion files. The source checks passed.'})
    return {'id': f'practice-{seed}', 'group': f'practice-{seed}', 'source': 'practice-handoff' if handoff else 'practice',
            'messages': messages, 'tools': TOOLS, 'tool_calls': calls,
            'construction': 'Original simulated file tools; exact target comparison, not executed Python tests',
            'verified_transformations': files_count, 'recovered_errors': failure_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=3000)
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument('--handoffs', action='store_true')
    args = parser.parse_args()
    histogram = {}
    with args.out.open('x') as stream:
        for index in range(args.count):
            record = episode(index + args.start_seed, (1, 2, 4, 8, 16)[index % 5],
                             padding=32 if index % 11 == 0 else 0, handoff=args.handoffs)
            histogram[str(record['tool_calls'])] = histogram.get(str(record['tool_calls']), 0) + 1
            stream.write(json.dumps(record) + '\n')
    with args.out.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    args.out.with_suffix('.source.json').write_text(json.dumps({'construction': 'Original deterministic simulation',
        'episodes': args.count, 'calls_histogram': histogram, 'sha256': checksum}, indent=2) + '\n')
    print(json.dumps(histogram))


if __name__ == '__main__':
    main()
