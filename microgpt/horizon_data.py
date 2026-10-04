"""Original evolving-file sessions with cumulative requirements and fresh handoffs.

Shares the arithmetic domain with the procedural horizon evaluation, but never
its seed or task instances. This measures template transfer, not broad agency.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from .protocol import reasoning_instruction
from .grounding import vocabulary
from .opencode_format import system_prompt, read_output


def sessions(seed, tools, rounds=8, words=None, wire=False, effort=None, recover_errors=False):
    rng = random.Random(seed)
    path = rng.choice(['solution.py', 'helpers.py', 'src/convert.py'])
    state_path = 'TASK_STATE.md'
    text = f"# {rng.choice(['Keep', 'Preserve', 'Retain'])} {rng.choice(['existing', 'earlier', 'previous'])} {rng.choice(['utilities', 'functions', 'calculations', 'helpers'])} {rng.choice(['when extending this module', 'while adding new code', 'and add the requested functions'])}.\n"
    if words and seed % 2:
        text = '# ' + ' '.join(rng.sample(words, rng.randrange(3, 10))) + '.\n'
    effort = effort or ('none', 'low', 'medium', 'high')[seed % 4]
    root = f'/tmp/project-{seed}/work'
    system = [{'role': 'system', 'content': reasoning_instruction(effort)},
              {'role': 'system', 'content': 'Read task state and the source before editing. Add the requested function without changing existing functions. Run check and retry transient failures. Report only observed results.'}]
    if wire:
        system[1]['content'] = system_prompt(root)
    messages = list(system)
    requirements = []
    checkpoints = []
    call_count = 0
    def action(name, arguments, output, thought, supervise=True):
        nonlocal call_count
        call_count += 1
        cid = f'call_{seed}_{call_count}'
        message = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': cid, 'type': 'function',
                    'function': {'name': name, 'arguments': arguments}}]}
        if effort != 'none':
            message['reasoning_content'] = thought
        if not supervise:
            message['supervise'] = False
        messages.extend([message, {'role': 'tool', 'tool_call_id': cid, 'content': output}])
    def read(file, content):
        numbered = '\n'.join(f'{i}: {line}' for i, line in enumerate(content.splitlines(), 1))
        display = '/tmp/project/work/' + file if not file.startswith('/') else file
        output = read_output(file, content, root) if wire else f'<path>{display}</path>\n<type>file</type>\n<content>\n{numbered}\n\n(End of file - total {len(content.splitlines())} lines)\n</content>'
        action('read', {'filePath': file}, output, 'I will inspect the current requirements and source.')
    def record(kind):
        return {'group': f'horizon-{seed}', 'source': 'horizon-' + kind, 'tools': tools,
                'messages': list(messages), 'fixture': {'source': text, 'requirements': list(requirements)},
                'construction': 'Original simulated tools; cumulative source assertions, held-out seeds'}
    for index in range(rounds):
        if index == rounds // 2:
            checkpoints.append(record('session'))
            messages = list(system)
        prefix = rng.choice(words or ['compute', 'transform', 'adjust', 'convert', 'scale', 'measure', 'shift', 'process', 'calculate', 'apply'])
        name = f'{prefix}_{rng.randrange(10000, 99999)}'
        while any(r['name'] == name for r in requirements):
            name = f'{prefix}_{rng.randrange(10000, 99999)}'
        factor, offset = rng.randrange(2, 30), rng.randrange(-50, 50)
        requirements.append({'name': name, 'factor': factor, 'offset': offset})
        notes = '# Requirements\n' + '\n'.join(f"- {r['name']}(value) must return value * {r['factor']} + {r['offset']}." for r in requirements)
        prompt = f'In {path} add {name}(value), returning value * {factor} + {offset}. Preserve all earlier functions. Read {state_path}, read the code, edit it, and call check. Retry transient errors.'
        if index == rounds // 2:
            prompt = 'Continue after a session handoff. ' + prompt
        messages.append({'role': 'user', 'content': json.dumps(prompt) if wire else prompt})
        read(state_path, notes)
        read(path, text)
        if seed % 5 == 0 and index == 0:
            # Recovery prefix: one redundant read must be followed by useful work.
            read(path, text)
        # Append by replacing the last observed function, keeping all earlier text.
        old = text[text.rfind('def '):] if 'def ' in text else text
        addition = f'def {name}(value):\n    return value * {factor} + {offset}\n'
        new = old.rstrip() + '\n\n' + addition
        assert text.count(old) == 1
        if recover_errors and index > 0 and index % 3 == 1 and seed % 2 == 0:
            # An off-policy mistaken edit is context, never a supervised target.
            duplicate = old.rstrip() + '\n\n' + old
            text = text.replace(old, duplicate, 1)
            action('edit', {'filePath': path, 'oldString': old, 'newString': duplicate},
                   'Edit applied successfully.', '', supervise=False)
            assert f'def {name}(value):' not in text
            action('check', {}, f"FAIL\nAttributeError: module 'solution' has no attribute '{name}'\n",
                   'I will check the changes.')
            # A stale single-function anchor is ambiguous after duplication.
            assert text.count(old) == 2
            action('edit', {'filePath': path, 'oldString': old, 'newString': new},
                   'Found multiple matches for oldString. Provide more surrounding context to make the match unique.',
                   '', supervise=False)
            read(path, text)
            old = duplicate
            assert text.count(old) == 1
        text = text.replace(old, new, 1)
        for r in requirements:
            assert f"def {r['name']}(value):\n    return value * {r['factor']} + {r['offset']}\n" in text
        action('edit', {'filePath': path, 'oldString': old, 'newString': new}, 'Edit applied successfully.', 'I will append the requested function and preserve the existing code.')
        if index == 2 and seed % 2 == 0:
            action('check', {}, 'TRANSIENT_ERROR: Test runner temporarily unavailable. Retry check.', 'I will test all current requirements.')
        action('check', {}, 'PASS\nPASS\n' if wire else 'PASS\n', 'I will verify all current requirements.')
        messages.append({'role': 'assistant', 'content': 'Added the function. All checks passed.'})
    checkpoints.append(record('handoff'))
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tools', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=1600)
    parser.add_argument('--vocabulary-data', type=Path, help='Existing training JSONL for diverse copied identifiers and comments')
    parser.add_argument('--initial-only', action='store_true', help='Supervise first-task decisions without diluting them in long histories')
    parser.add_argument('--wire', action='store_true', help='Match installed OpenCode environment and observation formatting')
    parser.add_argument('--effort', choices=['none', 'low', 'medium', 'high'])
    parser.add_argument('--recover-errors', action='store_true', help='Train recovery from mistaken edits without supervising those mistakes')
    args = parser.parse_args()
    tools = json.loads(args.tools.read_text())
    words = vocabulary(args.vocabulary_data) if args.vocabulary_data else None
    with args.out.open('x') as stream:
        for index in range(args.count):
            records = sessions(500000 + index, tools, rounds=2 if args.initial_only else (2, 4, 8, 16, 32)[index % 5], words=words, wire=args.wire, effort=args.effort, recover_errors=args.recover_errors)
            if args.initial_only:
                records = records[:1]
                records[0]['source'] = 'horizon-start'
            for item in records:
                stream.write(json.dumps(item) + '\n')
    with args.out.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    args.out.with_suffix('.source.json').write_text(json.dumps({'sessions': args.count, 'records': args.count * (1 if args.initial_only else 2),
        'seed_start': 500000, 'sha256': checksum, 'tools_sha256': hashlib.sha256(args.tools.read_bytes()).hexdigest(),
        'vocabulary_sha256': hashlib.sha256(json.dumps(words).encode()).hexdigest(), 'vocabulary_source': str(args.vocabulary_data),
        'wire': args.wire, 'effort': args.effort,
        'recover_errors': args.recover_errors,
        'note': 'Simulated demonstrations; whole episodes exceeding the token limit are excluded during preparation.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
