"""Make varied tool demonstrations from public educational code, checked in Docker.

References and source tests execute only inside a restricted networkless container.
The resulting traces are simulated demonstrations, not rollouts of our model.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import uuid

from microgpt.curate import rows, normalize_reference_tests
from microgpt.opencode_format import system_prompt, read_output
from microgpt.programming_data import GENERAL_PROMPT
from microgpt.protocol import reasoning_instruction
from eval_opencode import IMAGE

WORKER = r'''
import json, subprocess, sys
for line in open('/input.jsonl'):
    row = json.loads(line)
    try:
        run = subprocess.run([sys.executable, '-I', '-B', '-c', row['code'] + '\n' + '\n'.join(row['testcase'])],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1)
        ok = run.returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    print(json.dumps({'id': row['seq_id'], 'passed': ok}), flush=True)
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--tools', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=4000)
    p.add_argument('--deduplicate-code', action='store_true',
                   help='Keep only the first occurrence of each Python AST before checking references')
    args = p.parse_args()
    if args.out.exists():
        p.error('Use a fresh output')
    selected = []
    seen = set()
    for r in rows(args.input):
        if not r.get('testcase') or len(r['code']) > 2500 or len(r['instruction']) > 1500:
            continue
        # Tests must start at module scope, not accidentally become an unreachable
        # continuation of the reference function's body.
        try:
            r = {**r, 'testcase': normalize_reference_tests(r['testcase'])}
        except (SyntaxError, ValueError):
            continue
        if len(r['testcase']) > 20 or sum(map(len, r['testcase'])) > 4000:
            continue
        if args.deduplicate_code:
            try:
                fingerprint = hashlib.sha256(ast.dump(ast.parse(r['code'])).encode()).hexdigest()
            except SyntaxError:
                continue
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
        selected.append(r)
        if len(selected) >= args.count:
            break
    with tempfile.TemporaryDirectory(prefix='microgpt-reference-') as directory:
        base = Path(directory); base.chmod(0o755)
        source = base / 'input.jsonl'; worker = base / 'worker.py'
        source.write_text(''.join(json.dumps(r) + '\n' for r in selected)); source.chmod(0o644)
        worker.write_text(WORKER); worker.chmod(0o644)
        name = 'microgpt-reference-' + uuid.uuid4().hex
        command = ['docker', 'run', '--rm', '--name', name, '--network', 'none', '--read-only', '--cap-drop', 'ALL',
                   '--security-opt', 'no-new-privileges', '--pids-limit', '64', '--memory', '512m', '--cpus', '2',
                   '--user', '65534:65534', '--tmpfs', '/tmp:rw,noexec,size=16m',
                   '--mount', f'type=bind,src={source},dst=/input.jsonl,readonly',
                   '--mount', f'type=bind,src={worker},dst=/worker.py,readonly', IMAGE, 'python', '-I', '-B', '/worker.py']
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=max(120, args.count * 2), check=True)
        finally:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
        passed = {r['id'] for line in run.stdout.splitlines() if (r := json.loads(line))['passed']}
    tools = json.loads(args.tools.read_text())
    count = 0
    with args.out.open('x') as stream:
        for r in selected:
            if r['seq_id'] not in passed:
                continue
            root = f'/tmp/project-{r["seq_id"]}/work'
            path = 'solution.py'
            before = '# Python utilities\n'
            code = before + '\n' + r['code'].strip() + '\n'
            messages = [{'role': 'system', 'content': reasoning_instruction('none')},
                        {'role': 'system', 'content': system_prompt(root, agent_prompt=GENERAL_PROMPT)},
                        {'role': 'user', 'content': json.dumps(r['instruction'] + ' Implement this in solution.py and run check.')}]
            for name, arguments, result in [('read', {'filePath': path}, read_output(path, before, root)),
                ('edit', {'filePath': path, 'oldString': before, 'newString': code}, 'Edit applied successfully.'),
                ('check', {}, 'PASS\nPASS\n')]:
                call = 'call_' + str(len(messages))
                messages += [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': call, 'type': 'function', 'function': {'name': name, 'arguments': arguments}}]},
                             {'role': 'tool', 'tool_call_id': call, 'content': result}]
            messages.append({'role': 'assistant', 'content': 'Implemented the function in solution.py. The supplied tests passed.'})
            record = {'group': hashlib.sha256(r['instruction'].encode()).hexdigest(), 'source': 'educational-tools',
                'messages': messages, 'tools': tools, 'upstream_id': r['seq_id'], 'license': 'MIT',
                'construction': 'Simulated tools; reference code passed source tests in restricted Docker. Source tests may be incomplete.'}
            stream.write(json.dumps(record) + '\n'); count += 1
    args.out.with_suffix('.source.json').write_text(json.dumps({'selected': len(selected), 'passed': count, 'image': IMAGE,
        'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(), 'dataset': 'OpenCoder-LLM/opc-sft-stage2',
        'revision': '7d28f40d579edd7c24402d17d0c7639f991e6f8d',
        'deduplicate_code': args.deduplicate_code,
        'selection': 'First eligible references in source order; not a representative random sample.',
        'output_sha256': hashlib.sha256(args.out.read_bytes()).hexdigest()}, indent=2)+'\n')
    print(f'Reference tests passed: {count}/{len(selected)}', flush=True)


if __name__ == '__main__':
    main()
