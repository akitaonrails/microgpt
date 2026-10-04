"""Small original exercises for copying requested paths and making verified edits.

The tools are simulated. Fixed arithmetic templates have executable unit tests;
this data does not include the repository-repair evaluation's four tasks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import keyword

from .practice import TOOLS
from .protocol import reasoning_instruction


def vocabulary(path, limit=10000):
    """Identifier/text diversity from existing training data, never executed."""
    words = set()
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            for message in record['messages']:
                words.update(w for w in re.findall(r'\b[A-Za-z][A-Za-z_]{2,20}\b', message.get('content') or '') if not keyword.iskeyword(w))
            if len(words) >= limit:
                break
    return sorted(words)[:limit]


def episode(seed, tools, read_only=False):
    rng = random.Random(seed)
    filename = rng.choice(['solution', 'main', 'helpers', 'convert', 'utils', 'numbers', 'transform']) + '.py'
    directory = rng.choice(['', 'src/', 'lib/', '/workspace/project/', '/tmp/project/work/', f'/tmp/project-{rng.randrange(10000)}/'])
    path = directory + filename
    name = rng.choice(['scale', 'adjust', 'transform', 'calculate', 'convert', 'compute'])
    factor, offset = rng.randint(2, 30), rng.randint(1, 40)
    expression = rng.choice([f'value * {factor} + {offset}', f'value + {offset}', f'value - {offset}', f'value * {factor}'])
    wrong = expression.replace('+', '-') if '+' in expression else expression.replace('-', '+') if '-' in expression else expression.replace('*', '+')
    before = f'def {name}(value):\n    return {wrong}\n'
    after = f'def {name}(value):\n    return {expression}\n'
    effort = 'low' if seed % 4 == 0 else 'none'
    messages = [{'role': 'system', 'content': reasoning_instruction(effort)},
                {'role': 'system', 'content': 'You fix Python code. Read the requested file, edit it to satisfy the request, then call check. Copy file paths from the request. Use exact source text without displayed line numbers. Correct failures. Give a brief final answer.'},
                {'role': 'user', 'content': (rng.choice([f'Read {path}.', f'Open {path} and tell me its function name.', f'Inspect the file {path}. Do not edit it.']) if read_only else
                    f'Fix {name}(value) in {path}. It must return {expression}. Use read, edit, and check tools.')}]
    def action(tool, arguments, result, thought):
        call_id = f'call_{seed}_{len(messages)}'
        message = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': call_id, 'type': 'function',
                    'function': {'name': tool, 'arguments': arguments}}]}
        if effort != 'none':
            message['reasoning_content'] = thought
        messages.extend([message, {'role': 'tool', 'tool_call_id': call_id, 'content': result}])
    numbered = '\n'.join(f'{i}: {line}' for i, line in enumerate(before.splitlines(), 1))
    action('read', {'filePath': path}, f'<path>{path}</path>\n<type>file</type>\n<content>\n{numbered}\n</content>', 'I will read the requested file.')
    if not read_only:
        old, new = f'return {wrong}', f'return {expression}'
        assert before.count(old) == 1 and before.replace(old, new, 1) == after
        action('edit', {'filePath': path, 'oldString': old, 'newString': new}, 'Edit applied successfully.', 'I will correct the expression using the observed source.')
        if seed % 7 == 0:
            action('check', {}, 'TRANSIENT_ERROR: Test runner temporarily unavailable. Retry check.', 'I will run the tests.')
        action('check', {}, 'PASS\n', 'I will verify the result.')
    messages.append({'role': 'assistant', 'content': f'The file defines {name}(value).' if read_only else 'Updated the function. Tests passed.'})
    return {'group': f'grounding-{seed}', 'source': 'grounding-read' if read_only else 'grounding-edit', 'messages': messages,
            'tools': tools, 'construction': 'Original simulated tools; exact edit verified',
            'fixture': {'path': path, 'before': before, 'after': after, 'name': name, 'expression': expression}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tools', type=Path, required=True, help='Installed OpenCode schemas from capture_opencode_tools.py')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=8000)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('count must be positive')
    native = json.loads(args.tools.read_text())
    with args.out.open('x') as stream:
        for index in range(args.count):
            # Mix concise and real harness schemas so neither becomes mandatory.
            tools = native if index % 2 else TOOLS[:3]
            item = episode(400000 + index, tools, read_only=index % 3 == 0)
            stream.write(json.dumps(item) + '\n')
    with args.out.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    args.out.with_suffix('.source.json').write_text(json.dumps({'count': args.count, 'sha256': checksum,
        'tools_sha256': hashlib.sha256(args.tools.read_bytes()).hexdigest(), 'seed_start': 400000,
        'note': 'Simulated training exercises, not agent evaluation results.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
