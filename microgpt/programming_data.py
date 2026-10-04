"""Programming instruction/tool lessons plus a bounded public coding mixture.

Only the authored templates below are executed by tests. Imported answers remain
text, never executable training fixtures. Evaluation prompts live separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from .opencode_format import read_output, system_prompt
from .protocol import reasoning_instruction

GENERAL_PROMPT = ('You are a helpful assistant. Follow the latest user request and preserve relevant earlier requirements. '
    'Answer questions directly. Use tools only when needed and only those offered. For code changes, inspect the actual file before editing, '
    'copy its path and source exactly, then check the result. Treat line numbers as annotations. '
    'Recover from errors using the observation, not by repeating a failed action. Ask a specific question when essential information is missing. '
    'Do not claim to have changed files or run tests unless a tool confirms it.')


def task(seed):
    rng = random.Random(seed)
    name = rng.choice(['convert', 'process', 'transform', 'compute', 'apply']) + '_' + str(seed % 997)
    n = rng.randint(2, 9)
    # Each entry is (signature, request, body, independent input/output examples).
    choices = [
        ('text', 'return the text in uppercase', 'return text.upper()', [('Abc', 'ABC'), ('', '')]),
        ('text', 'remove leading and trailing whitespace', 'return text.strip()', [(' a  ', 'a'), ('', '')]),
        ('text', 'return the characters in reverse order', 'return text[::-1]', [('abc', 'cba'), ('', '')]),
        ('values', 'return the number of items', 'return len(values)', [([1, 2, 3], 3), ([], 0)]),
        ('values', 'return the sum of the numbers', 'return sum(values)', [([1, -2, 3], 2), ([], 0)]),
        ('values', 'return the items in ascending order without changing the input', 'return sorted(values)', [([3, 1, 2], [1, 2, 3]), ([], [])]),
        ('value', f'return the value raised to the power {n}', f'return value ** {n}', [(2, 2**n), (0, 0)]),
        ('value', 'return whether the integer is even', 'return value % 2 == 0', [(4, True), (-3, False)]),
        ('values', 'return only strictly positive numbers', 'return [value for value in values if value > 0]', [([-2, 0, 3], [3]), ([], [])]),
        ('values', f'return a list with each number multiplied by {n}', f'return [value * {n} for value in values]', [([1, -2], [n, -2*n]), ([], [])]),
        ('text', f'return the text repeated {n} times', f'return text * {n}', [('a', 'a'*n), ('', '')]),
        ('text', 'return whether the text is a palindrome', 'return text == text[::-1]', [('aba', True), ('ab', False)]),
        ('values', 'return the first item or None when the list is empty', 'return values[0] if values else None', [([4, 9], 4), ([], None)]),
        ('text', 'split the text at whitespace into words', 'return text.split()', [('a  b\nc', ['a', 'b', 'c']), (' ', [])]),
        ('text', 'replace spaces with underscores', "return text.replace(' ', '_')", [('a b', 'a_b'), ('', '')]),
        ('name', 'return a welcome message in the form Welcome, NAME!', 'return f"Welcome, {name}!"', [('Ada', 'Welcome, Ada!'), ('João', 'Welcome, João!')]),
        ('name', 'print a welcome message in the form Welcome, NAME! and return None', 'print(f"Welcome, {name}!")', []),
        ('value', f'add {n} to the number', f'return value + {n}', [(3, 3+n), (-n, 0)]),
    ]
    family = rng.randrange(len(choices))
    signature, request, body, cases = choices[family]
    code = f'def {name}({signature}):\n    {body}\n'
    return {'family': family, 'name': name, 'signature': signature, 'request': request, 'code': code, 'cases': cases}


def episode(seed, tools):
    rng = random.Random(seed + 31)
    mode = seed % 4
    root = f'/tmp/project-{seed}/work'
    path = rng.choice(['solution.py', 'main.py', 'helpers.py', 'src/utils.py'])
    wire = seed % 3 != 0
    messages = [{'role': 'system', 'content': reasoning_instruction('none')},
                {'role': 'system', 'content': system_prompt(root, agent_prompt=GENERAL_PROMPT) if wire else GENERAL_PROMPT}]
    fixtures = []
    content = '# Python utilities\n'
    def user(text):
        messages.append({'role': 'user', 'content': json.dumps(text) if wire else text})
    def action(name, args, result, supervise=True):
        call_id = f'call_{len(messages)}'
        messages.extend([{'role': 'assistant', 'content': '', 'supervise': supervise,
            'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': args}}]},
            {'role': 'tool', 'tool_call_id': call_id, 'content': result}])
    for turn in range(1 if mode == 0 else 3):
        item = task(seed * 7 + turn * 100003)
        fixtures.append(item)
        prompt = rng.choice([f"Write {item['name']}({item['signature']}) to {item['request']}.",
                             f"I need a Python function named {item['name']}({item['signature']}). It should {item['request']}.",
                             f"Implement {item['name']}({item['signature']}): {item['request']}."])
        if mode == 0:
            user(prompt + ' Show the code only; do not edit files.')
            messages.append({'role': 'assistant', 'content': '```python\n' + item['code'] + '```'})
            continue
        user(prompt + f' Add it to {path}.' + (' Keep the earlier functions.' if turn else ''))
        if mode == 3 and turn == 1:
            action('read', {'filePath': 'missing.py'}, 'Error: File not found: missing.py', supervise=False)
        action('read', {'filePath': path}, read_output(path, content, root))
        updated = content + '\n' + item['code']
        action('edit', {'filePath': path, 'oldString': content, 'newString': updated}, 'Edit applied successfully.')
        if mode == 3 and turn == 2:
            action('check', {}, 'TRANSIENT_ERROR: Test runner temporarily unavailable. Retry check.')
        action('check', {}, 'PASS\nPASS\n')
        messages.append({'role': 'assistant', 'content': f'Added {item["name"]} to {path}. Tests passed.'})
        content = updated
        if mode == 2 and turn == 0:
            user('What is the difference between printing and returning? Do not change files.')
            messages.append({'role': 'assistant', 'content': 'Printing writes text to the output stream. Returning passes a value back to the caller. A function can do both; without an explicit return value, it returns None.'})
    return {'group': f'general-original-{seed}', 'source': 'programming-direct' if mode == 0 else 'programming-tools',
            'tools': tools if mode else [], 'messages': messages, 'fixtures': fixtures,
            'construction': 'Authored templates and simulated tool results; not real agent success.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tools', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--public', type=Path, nargs='+', required=True)
    parser.add_argument('--count', type=int, default=8000)
    parser.add_argument('--per-source', type=int, default=100000)
    parser.add_argument('--max-chars', type=int, default=16000)
    args = parser.parse_args()
    tools = json.loads(args.tools.read_text())
    stats, sources = {}, []
    with args.out.open('x') as target:
        def emit(item):
            target.write(json.dumps(item, ensure_ascii=False) + '\n')
            stats[item['source']] = stats.get(item['source'], 0) + 1
        for seed in range(910000, 910000 + args.count):
            emit(episode(seed, tools))
        for path in args.public:
            sources.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            count = 0
            for line in path.open():
                record = json.loads(line)
                # Short complete conversations, never truncation through an answer.
                if sum(len(m.get('content') or '') for m in record['messages']) > args.max_chars:
                    continue
                if record.get('tools') or any(m.get('tool_calls') for m in record['messages']):
                    continue
                # Direct answers and tool examples must share system-prompt variants:
                # otherwise the model can learn 'this system prompt means edit'.
                variant = int(hashlib.sha256(str(record['group']).encode()).hexdigest()[:8], 16) % 3
                if variant:
                    system = GENERAL_PROMPT if variant == 1 else 'You are a coding assistant. Follow the requested interface and answer directly when no tools are needed.'
                    record['messages'].insert(0, {'role': 'system', 'content': system})
                record['messages'].insert(0, {'role': 'system', 'content': reasoning_instruction('none')})
                emit(record)
                count += 1
                if count >= args.per_source:
                    break
    args.out.with_suffix('.source.json').write_text(json.dumps({'counts': stats, 'public_sources': sources,
        'sha256': hashlib.sha256(args.out.read_bytes()).hexdigest(),
        'note': 'Public source provenance retained; imported code is never executed.'}, indent=2) + '\n')
    print(stats)


if __name__ == '__main__':
    main()
