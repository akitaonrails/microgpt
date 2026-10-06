"""Vary request formatting while preserving checked code and split groups.

Uses only the public training curriculum. No benchmark prompts or answers are
read. Signature information remains explicit in every format.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path


def request(instruction, name, parameters, style):
    signature = f'{name}({parameters})'
    variants = [
        f'{instruction} Use the exact function signature {signature}.',
        f'Implement {signature}. {instruction}',
        f'Write a Python function {signature}. {instruction}',
        f'I need a function with signature {signature}. {instruction}',
        f'{instruction}\n\nRequired interface:\n```python\ndef {signature}:\n    ...\n```',
        f'Task: {instruction}\nName and parameters: {signature}',
        f'Please provide an implementation of {signature} that meets this specification:\n{instruction}',
        f'{instruction}\nPut the implementation in a function called {name} with parameters {parameters}.',
    ]
    return variants[style] + ' Return only Python code in a fenced block.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a fresh output')
    occurrences = Counter()
    styles = Counter()
    with args.out.open('x') as output:
        for line in args.input.open():
            record = json.loads(line)
            code = record['messages'][-1]['content'].removeprefix('```python\n').removesuffix('```')
            function = next(node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef))
            parameters = ast.unparse(function.args)
            user = next(message for message in record['messages'] if message['role'] == 'user')
            suffix = (f' Use the exact function signature {function.name}({parameters}).'
                      ' Return only Python code in a fenced block.')
            if not user['content'].endswith(suffix):
                raise ValueError(f'Unexpected source prompt for {record["upstream_id"]}')
            identifier = str(record['upstream_id'])
            offset = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16) % 8
            style = (offset + occurrences[identifier]) % 8
            user['content'] = request(user['content'][:-len(suffix)], function.name, parameters, style)
            occurrences[identifier] += 1
            styles[style] += 1
            record['construction'] = 'Checked code preserved; only request formatting varied. Source-linked split groups preserved.'
            output.write(json.dumps(record) + '\n')
    metadata = {'source_records': len(occurrences), 'episodes': sum(styles.values()),
                'styles': dict(styles), 'changed_fields': ['user content', 'construction'],
                'note': 'No change to assistant code, system messages, tools, source IDs, or split groups.'}
    for key, path in [('input_sha256', args.input), ('output_sha256', args.out)]:
        with path.open('rb') as stream:
            metadata[key] = hashlib.file_digest(stream, 'sha256').hexdigest()
    args.out.with_suffix('.source.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
