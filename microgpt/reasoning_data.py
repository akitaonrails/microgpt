"""Import complete public worked solutions and build checkable program traces.

Public traces retain their source validation status; we do not re-run arbitrary
downloaded programs. Original trace exercises have deterministic numeric oracles.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .curate import rows
from .protocol import reasoning_instruction, body
from .tokenizer import Tokenizer
from .trajectories import assistant_examples


def trace_problem(seed, family=None):
    rng = random.Random(seed)
    family = seed % 3 if family is None else family
    values = [rng.randint(-9, 14) for _ in range(rng.randint(4, 10))]
    initial = rng.randint(-10, 10)
    steps = []
    if family == 0:
        factor = rng.randint(2, 5)
        code = f'values = {values!r}\ntotal = {initial}\nfor value in values:\n    total += value * {factor}\nprint(total)'
        current = initial
        for value in values:
            before = current
            current += value * factor
            steps.append(f'For {value}: {before} + ({value} * {factor}) = {current}.')
        expected = initial + factor * sum(values)
    elif family == 1:
        divisor = rng.randint(2, 4)
        code = f'values = {values!r}\nscore = {initial}\nfor value in values:\n    if value % {divisor} == 0:\n        score += value\n    else:\n        score -= 1\nprint(score)'
        current = initial
        for value in values:
            divisible = value % divisor == 0
            current += value if divisible else -1
            steps.append(f'{value} is {"" if divisible else "not "}divisible by {divisor}; score becomes {current}.')
        expected = initial + sum(x for x in values if x % divisor == 0) - sum(x % divisor != 0 for x in values)
    elif family == 2:
        take = rng.randint(1, len(values) - 1)
        code = f'values = {values!r}\nselected = sorted(values)[:{take}]\nresult = sum(selected) - min(values)\nprint(result)'
        ordered = sorted(values)
        current = sum(ordered[:take]) - ordered[0]
        steps = [f'The sorted list is {ordered}.', f'The first {take} values are {ordered[:take]}.',
                 f'Their sum is {sum(ordered[:take])}. Subtracting the minimum {ordered[0]} gives {current}.']
        expected = sum(sorted(values)[1:take])
    elif family == 3:
        # Reserved from procedural TRAINING: evaluation tests a new template.
        width = rng.randint(2, 6)
        code = f'values = {values!r}\ncount = 0\nfor value in values:\n    for offset in range({width}):\n        if value + offset > 0:\n            count += 1\nprint(count)'
        current = 0
        for value in values:
            hits = sum(value + i > 0 for i in range(width))
            current += hits
            steps.append(f'For {value}, {hits} offsets satisfy the condition; count becomes {current}.')
        expected = sum(max(0, min(width, width + x - 1)) for x in values)
    else:
        raise ValueError('Unknown trace family')
    assert current == expected
    question = 'What integer does this Python program print? Return only that integer as your final answer.\n```python\n' + code + '\n```'
    thought = 'I will track the state in execution order.\n' + '\n'.join(steps) + f'\nThe final printed integer is {expected}.'
    return {'question': question, 'reasoning': thought, 'answer': str(expected), 'family': family, 'code': code}


def record(question, thought, answer, source, effort='medium'):
    group = hashlib.sha256(question.encode()).hexdigest()
    response = {'role': 'assistant', 'content': answer}
    if effort != 'none':
        response['reasoning_content'] = thought
    return {'group': group, 'source': source, 'tools': [], 'messages': [
        {'role': 'system', 'content': reasoning_instruction(effort)},
        {'role': 'user', 'content': question}, response]}


def import_public(directory, source, tokenizer, max_length, limit, counts):
    manifest = json.loads((directory / 'manifest.json').read_text())
    seen = set()
    for entry in manifest['files']:
        path = (directory / entry['path']).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError('Source path escapes the dataset directory')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != entry['sha256']:
                raise ValueError('Changed reasoning source')
        for row in rows(path):
            counts['scanned'] += 1
            messages = row['messages']
            if len(messages) != 2 or [m['role'] for m in messages] != ['user', 'assistant']:
                counts['unsupported_roles'] += 1
                continue
            text = messages[1]['content'].strip()
            if not text.startswith('<think>') or text.count('</think>') != 1:
                counts['incomplete_reasoning'] += 1
                continue
            thought, answer = text[len('<think>'):].split('</think>', 1)
            if not thought.strip() or not answer.strip():
                counts['empty_component'] += 1
                continue
            if source == 'reasoning-code' and '```python' not in answer.lower():
                counts['not_python'] += 1
                continue
            question = messages[0]['content']
            group = hashlib.sha256(question.encode()).hexdigest()
            if group in seen:
                counts['duplicate_problem'] += 1
                continue
            seen.add(group)
            thought_tokens = len(tokenizer.encode(thought))
            effort = 'low' if thought_tokens <= 256 else 'medium' if thought_tokens <= 1024 else 'high'
            item = record(question, thought.strip(), answer.strip(), source, effort)
            if len(tokenizer.encode(body(item['messages'][-1]))) + 1 > 4096:
                counts['over_output_budget'] += 1
                continue
            if not assistant_examples(item, tokenizer, max_length, pack=True):
                counts['overlength'] += 1
                continue
            item.update(dataset=manifest['dataset'], revision=manifest['revision'], upstream=row['source'],
                        verification='Source authors report verified traces; not independently re-executed here')
            counts['accepted'] += 1
            yield item
            if counts['accepted'] >= limit:
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['reasoning-code', 'reasoning-math', 'reasoning-trace', 'reasoning-tools'], required=True)
    parser.add_argument('--input', type=Path, nargs='+', help='Original practice JSONL for thinking/tool variants')
    parser.add_argument('--raw', type=Path, default=Path('/mnt/gigachad/microgpt/dataset/raw'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=6000)
    parser.add_argument('--max-length', type=int, default=8192)
    args = parser.parse_args()
    if args.out.exists() or args.limit < 1:
        parser.error('Choose a fresh output file and a positive limit')
    counts = Counter()
    tokenizer = Tokenizer(encoding='o200k_base')
    def synthetic():
        for index in range(args.limit):
            problem = trace_problem(200000 + index // 2)
            # Paired direct/reasoned variants share a problem group and split.
            effort = 'none' if index % 2 == 0 else ('low', 'medium', 'high')[(index // 2) % 3]
            counts['accepted'] += 1
            yield {**record(problem['question'], problem['reasoning'], problem['answer'], args.source, effort),
                   'verification': 'Original bounded Python templates; deterministic numeric oracle', 'family': problem['family']}
    def tools():
        if not args.input:
            raise ValueError('reasoning-tools requires --input practice files')
        for path in args.input:
            with path.open() as stream:
                for line in stream:
                    item = json.loads(line)
                    if item['source'] not in ('practice', 'practice-handoff'):
                        raise ValueError('Only original simulated exercises are supported here')
                    item['source'] = 'reasoning-tools'
                    item['messages'].insert(0, {'role': 'system', 'content': reasoning_instruction('medium')})
                    for message in item['messages']:
                        if message.get('tool_calls'):
                            name = message['tool_calls'][0]['function']['name']
                            plans = {'read': 'I will inspect the current file.', 'edit': 'I will apply the requested replacement using the observed source.',
                                     'check': 'I will check the current changes before reporting the result.', 'todowrite': 'I will record completed and remaining work.'}
                            message['reasoning_content'] = message.get('content') or plans[name]
                            message['content'] = ''
                    if not assistant_examples(item, tokenizer, args.max_length, pack=True):
                        counts['overlength'] += 1
                        continue
                    counts['accepted'] += 1
                    yield item
                    if counts['accepted'] >= args.limit:
                        return
    if args.source == 'reasoning-trace':
        iterator = synthetic()
    elif args.source == 'reasoning-tools':
        iterator = tools()
    else:
        iterator = import_public(args.raw / args.source, args.source, tokenizer, args.max_length, args.limit, counts)
    temporary = args.out.with_suffix('.partial')
    with temporary.open('w') as stream:
        for item in iterator:
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')
    if not counts['accepted']:
        raise ValueError(f'No usable reasoning records: {counts}')
    temporary.replace(args.out)
    with args.out.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {'source': args.source, 'counts': dict(counts), 'sha256': checksum,
              'max_length': args.max_length, 'note': 'Whole public examples are filtered, never truncated. No model weights loaded.'}
    args.out.with_suffix('.source.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
