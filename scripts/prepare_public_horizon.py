"""Compose checked public programs into executed cumulative file-edit lessons.

Reference-policy demonstrations, not model rollouts. Each new implementation must
change a failing cumulative check into a passing one. Source split membership is
preserved; the harness maintains a requirements file containing no answers.
"""
import argparse
import ast
from collections import Counter
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
from microgpt.tokenizer import Tokenizer
from microgpt.trajectories import assistant_examples
from prepare_verified_code import checked_references
from prepare_executed_repairs import random_imports, scoped_edit
from eval_opencode import IMAGE


def split_for(group):
    return 'val' if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 20 == 0 else 'train'


def select_session(records, seed, rounds):
    group = f'public-horizon-{seed}'
    pool = [r for r in records if split_for(r['group']) == split_for(group)]
    random.Random(seed).shuffle(pool)
    chosen, bindings = [], {}
    for record in pool:
        additions = {}
        for node in ast.parse(record['code']).body:
            if isinstance(node, ast.FunctionDef):
                additions[node.name] = ('function', record['seq_id'])
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or (alias.name.split('.')[0] if isinstance(node, ast.Import) else alias.name)
                    additions[name] = ('import', ast.dump(node))
        if '*' in additions or any(name in bindings and bindings[name] != value
                                   for name, value in additions.items()):
            continue
        # A source may occur only once in a session, even if its function is identical.
        if any(r['seq_id'] == record['seq_id'] or r['group'] == record['group'] for r in chosen):
            continue
        chosen.append(record)
        bindings.update(additions)
        if len(chosen) == rounds:
            return {'group': group, 'records': chosen, 'seed': seed}
    raise ValueError(f'Not enough noncolliding {split_for(group)} programs for {rounds} rounds')


WORKER = inspect.getsource(scoped_edit) + r'''
import json, pathlib, subprocess, sys, tempfile

def check(root, records):
    # Each task's tests get a fresh namespace so test variables cannot mask
    # another task's globals. Every namespace loads the entire current file.
    text = 'source = open("solution.py").read()\n'
    for record in records:
        text += 'namespace = {}\nexec(compile(source, "solution.py", "exec"), namespace)\n'
        text += 'exec(' + repr('\n'.join(record['testcase'])) + ', namespace)\n'
    (root / 'check.py').write_text(text)
    try:
        run = subprocess.run([sys.executable, '-I', '-B', 'check.py'], cwd=root,
                             capture_output=True, text=True, timeout=5)
        return {'returncode': run.returncode, 'output': (run.stdout + run.stderr)[-6000:]}
    except subprocess.TimeoutExpired:
        return {'returncode': None, 'output': 'Cumulative check timed out.'}

for line in open('/input.jsonl'):
    item = json.loads(line)
    result = {'group': item['group'], 'retained': False, 'rounds': []}
    with tempfile.TemporaryDirectory(prefix='public-session-') as directory:
        root = pathlib.Path(directory)
        solution = root / 'solution.py'
        solution.write_text('# Preserve earlier functions when adding new ones.\n')
        for index, record in enumerate(item['records']):
            before = solution.read_text()
            previous = check(root, item['records'][:index + 1])
            if previous['returncode'] is None or previous['returncode'] == 0:
                result['rejection'] = 'Missing implementation did not produce a timely failing check'
                break
            after = before.rstrip() + '\n\n' + record['code']
            old, new = scoped_edit(before, after)
            assert before.count(old) == 1
            solution.write_text(before.replace(old, new, 1))
            assert solution.read_text() == after
            current = check(root, item['records'][:index + 1])
            if current['returncode'] != 0:
                result['rejection'] = 'Combined reference failed or timed out'
                result['failure'] = current
                break
            result['rounds'].append({'before': before, 'after': after, 'old_string': old,
                'new_string': new, 'negative': previous, 'positive': current})
        else:
            result['retained'] = True
        result['root'] = str(root)
    print(json.dumps(result), flush=True)
'''


def episodes(session, checked, tools, handoff):
    root = checked['root']
    system = [{'role': 'system', 'content': reasoning_instruction('none')},
              {'role': 'system', 'content': system_prompt(root, GENERAL_PROMPT).replace(
                  'Is directory a git repo: yes', 'Is directory a git repo: no')}]
    messages = list(system)
    output = []
    requirements = []

    def record():
        return {'group': session['group'], 'source': 'executed-public-horizon', 'tools': tools,
                'messages': list(messages), 'upstream_ids': [r['seq_id'] for r in session['records']],
                'source_groups': [r['group'] for r in session['records']], 'license': 'CC-BY-4.0',
                'construction': 'Executed reference-policy cumulative edits; not a model or OpenCode rollout. Harness-written requirements, no reference answers in user/state messages.',
                'session_rounds': len(session['records']), 'handoff_every': handoff}

    def action(name, arguments, observation):
        identifier = f'call_{len(messages)}'
        messages.extend([{'role': 'assistant', 'content': '', 'tool_calls': [{'id': identifier,
            'type': 'function', 'function': {'name': name, 'arguments': arguments}}]},
            {'role': 'tool', 'tool_call_id': identifier, 'content': observation}])

    for index, (source, check) in enumerate(zip(session['records'], checked['rounds'], strict=True)):
        if handoff and index and index % handoff == 0:
            output.append(record())
            messages = list(system)
        requirements.append(source['prompt'] + '\nRequired signature: ' + source['signature'])
        notes = '# Required behavior\n\n' + '\n\n'.join(requirements) + '\n'
        messages.append({'role': 'user', 'content': json.dumps('In solution.py, implement the following request. '
            'Preserve earlier functions. Read TASK_STATE.md and the source, edit, then run check.\n' + requirements[-1])})
        action('read', {'filePath': 'TASK_STATE.md'}, read_output('TASK_STATE.md', notes, root))
        action('read', {'filePath': 'solution.py'}, read_output('solution.py', check['before'], root))
        action('edit', {'filePath': 'solution.py', 'oldString': check['old_string'],
                        'newString': check['new_string']}, 'Edit applied successfully.')
        action('check', {}, 'PASS\n' + check['positive']['output'])
        messages.append({'role': 'assistant', 'content': 'Added the function. The cumulative checks passed.'})
    output.append(record())
    return output


def fit_context(episode, tokenizer, limit):
    """Handoff only between complete user turns; never silently lose late tasks."""
    messages = episode['messages']
    starts = [i for i, message in enumerate(messages) if message['role'] == 'user']
    prefix = messages[:starts[0]]
    current = list(prefix)
    output = []
    for start, end in zip(starts, starts[1:] + [len(messages)], strict=True):
        turn = messages[start:end]
        candidate = {**episode, 'messages': current + turn}
        if not assistant_examples(candidate, tokenizer, limit, pack=True, canonical_tool_ids=True):
            if len(current) == len(prefix):
                raise ValueError('A single task exceeds the context limit')
            output.append({**episode, 'messages': current, 'context_limit': limit})
            current = list(prefix)
            candidate = {**episode, 'messages': current + turn}
            if not assistant_examples(candidate, tokenizer, limit, pack=True, canonical_tool_ids=True):
                raise ValueError('A single task exceeds the context limit')
        current = current + turn
    output.append({**episode, 'messages': current, 'context_limit': limit})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--curriculum', type=Path, required=True)
    parser.add_argument('--tools', type=Path, default=Path('data/opencode-tools-v2.json'))
    parser.add_argument('--tokenizer', type=Path, default=Path('data/large-v1'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-count', type=int, default=20000)
    parser.add_argument('--sessions', type=int, default=100)
    parser.add_argument('--rounds', type=int, default=32)
    parser.add_argument('--handoff', type=int, default=4, help='Fresh history every N rounds; zero disables scheduled handoffs, but the context limit still applies')
    parser.add_argument('--max-context', type=int, default=32768, help='Additional complete-turn handoffs when history exceeds this limit')
    args = parser.parse_args()
    if args.out.exists() or min(args.source_count, args.sessions, args.rounds, args.max_context) < 1 or args.handoff < 0:
        parser.error('Use a fresh output and positive counts')
    tokenizer = Tokenizer.load(args.tokenizer)
    source_manifest = json.loads((args.source / 'manifest.json').read_text())
    records = [r for r in checked_references(args.source, args.curriculum, args.source_count)
               if len(tokenizer.encode(r['code'])) <= 140
               and len(tokenizer.encode(r['prompt'])) <= 160
               and not random_imports(r['code'] + '\n' + '\n'.join(r['testcase']))]
    sessions = [select_session(records, 2026101000 + i, args.rounds) for i in range(args.sessions)]
    with tempfile.TemporaryDirectory(prefix='microgpt-public-horizon-') as directory:
        base = Path(directory); base.chmod(0o755)
        source = base / 'input.jsonl'; source.write_text(''.join(json.dumps(s) + '\n' for s in sessions)); source.chmod(0o644)
        worker = base / 'worker.py'; worker.write_text(WORKER); worker.chmod(0o644)
        name = 'microgpt-public-horizon-' + uuid.uuid4().hex
        command = ['docker', 'run', '--rm', '--name', name, '--network', 'none', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--pids-limit', '64', '--memory', '512m', '--cpus', '2',
            '--user', '65534:65534', '--tmpfs', '/tmp:rw,noexec,size=16m',
            '--mount', f'type=bind,src={source},dst=/input.jsonl,readonly',
            '--mount', f'type=bind,src={worker},dst=/worker.py,readonly', IMAGE, 'python', '-I', '-B', '/worker.py']
        try:
            run = subprocess.run(command, capture_output=True, text=True,
                                 timeout=max(600, args.sessions * args.rounds * 12), check=True)
        finally:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
    checks = [json.loads(line) for line in run.stdout.splitlines()]
    if [c['group'] for c in checks] != [s['group'] for s in sessions]:
        raise ValueError('Incomplete session checks')
    tools = json.loads(args.tools.read_text())
    counts = Counter()
    with args.out.open('x') as output:
        for session, checked in zip(sessions, checks, strict=True):
            if checked['retained']:
                for episode in episodes(session, checked, tools, args.handoff):
                    for fitted in fit_context(episode, tokenizer, args.max_context):
                        output.write(json.dumps(fitted) + '\n')
                        counts[split_for(episode['group'])] += 1
    args.out.with_suffix('.checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    report = {'dataset': source_manifest['dataset'], 'revision': source_manifest['revision'],
              'sessions_requested': args.sessions, 'sessions_retained': sum(c['retained'] for c in checks),
              'episodes_by_split': counts, 'rounds': args.rounds, 'handoff_every': args.handoff,
              'context_limit': args.max_context, 'additional_context_handoffs': True,
              'eligible_source_programs': len(records), 'image': IMAGE,
              'tools_sha256': hashlib.sha256(args.tools.read_bytes()).hexdigest(),
              'curriculum_sha256': hashlib.sha256(args.curriculum.read_bytes()).hexdigest(),
              'output_sha256': hashlib.sha256(args.out.read_bytes()).hexdigest(),
              'checks_sha256': hashlib.sha256(args.out.with_suffix('.checks.json').read_bytes()).hexdigest(),
              'note': 'Prepared data only. No learned long-horizon capability established. Exact split preservation does not prove semantic decontamination.'}
    args.out.with_suffix('.source.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
