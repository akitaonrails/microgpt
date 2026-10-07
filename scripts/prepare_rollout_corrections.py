"""Correct captured training-source failure states using executed references.

Prior model actions remain context with masked labels. Corrections are teacher
edits checked on the captured file state; they are not model successes.
"""
import argparse
import hashlib
import json
from pathlib import Path

from microgpt.chat import ChatRequest, normalize
from microgpt.opencode_format import read_output, cli_user_content
from prepare_executed_repairs import scoped_edit
from prepare_public_horizon import split_for
from recheck_public_code import sandbox


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--captures', type=Path, required=True)
    p.add_argument('--tasks-file', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    tasks = [json.loads(line) for line in args.tasks_file.open()]
    if any(split_for(t['group']) != 'train' for t in tasks):
        raise ValueError('Only training sources may supply corrective targets')
    prompts = {cli_user_content(t['prompt'] + ' Use read, edit, and check tools.'): t for t in tasks}
    if len(prompts) != len(tasks): raise ValueError('Training prompts must be unique')
    refs = {r['seq_id']: r for line in Path('data/structural-repairs-v2/references.jsonl').open() if (r := json.loads(line))}
    last = {}
    for line in args.captures.open():
        row = json.loads(line)
        messages, tools = normalize(ChatRequest(**row['request']))
        user = next(m['content'] for m in messages if m['role'] == 'user')
        if user not in prompts: raise ValueError('Captured prompt does not match a unique training fixture')
        task = prompts[user]
        last[task['id']] = {**row, 'messages': messages, 'tools': tools, 'task': task}
    candidates = {}; inputs = []
    for ident, row in last.items():
        reference = refs[ident.removeprefix('train_')]
        before = row['code']; after = reference['canonical'] + '\ndef _preserved_helper():\n    return "unchanged"\n'
        if not before or before == after: continue
        old, new = scoped_edit(before, after)
        if not old or before.count(old) != 1: continue
        row.update(old=old, new=new, after=after)
        candidates[ident] = row
        inputs.append({'seq_id': ident, 'code': after, 'empty': before,
            'testcase': reference['testcase'] + ['assert _preserved_helper() == "unchanged"']})
    source = args.out / 'controls-input.jsonl'
    source.write_text(''.join(json.dumps(r) + '\n' for r in inputs)); source.chmod(0o644)
    sandbox(source, args.out / 'controls.jsonl', [1, 2, 3], max(600, len(inputs) * 20))
    accepted = {r['id'] for line in (args.out / 'controls.jsonl').open() if (r := json.loads(line))['retained']}
    with (args.out / 'corrections.jsonl').open('x') as stream:
        for ident in sorted(accepted):
            row = candidates[ident]; messages = row['messages']
            for m in messages:
                if m['role'] == 'assistant': m['supervise'] = False
            def action(name, arguments, observation):
                call_id = f'correction_{len(messages)}'
                messages.extend([{'role': 'assistant', 'content': '', 'tool_calls': [{'id': call_id,
                    'type': 'function', 'function': {'name': name, 'arguments': arguments}}]},
                    {'role': 'tool', 'tool_call_id': call_id, 'content': observation}])
            action('read', {'filePath': 'solution.py'}, read_output('solution.py', row['code'], row['workdir']))
            action('edit', {'filePath': 'solution.py', 'oldString': row['old'], 'newString': row['new']}, 'Edit applied successfully.')
            action('check', {}, 'PASS\n')
            messages.append({'role': 'assistant', 'content': 'Corrected the implementation. The supplied checks passed.'})
            stream.write(json.dumps({'group': row['task']['group'], 'source': 'executed-public-repair',
                'upstream_id': ident.removeprefix('train_'), 'license': 'CC-BY-4.0',
                'messages': messages, 'tools': row['tools'],
                'construction': 'Captured model history and file state from actual OpenCode on a training source. Previous actions masked. Reference correction independently checked under three hash seeds; correction itself is not a model rollout.'}) + '\n')
    report = {'captured_tasks': len(last), 'candidate_failures': len(inputs), 'verified_corrections': len(accepted),
        'captures_sha256': hashlib.sha256(args.captures.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256((args.out / 'corrections.jsonl').read_bytes()).hexdigest(), 'seeds': [1, 2, 3]}
    (args.out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
