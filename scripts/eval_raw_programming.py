"""Raw code-completion diagnostic using the same sandboxed programming probes.

Supplies signature/docstring, not the reference body. This is a diagnostic format
comparison, not a contamination-free benchmark. Generated code runs only in Docker.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import uuid

import torch
from microgpt.inference import NativeBackend
from eval_programming import TASKS, REFERENCES
from eval_opencode import docker_command, grade, IMAGE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tasks-file', type=Path, help='Frozen JSONL task set; references are used only in assisted controls')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    records = [json.loads(line) for line in args.tasks_file.open()] if args.tasks_file else None
    tasks = [(r['id'], r['prompt'], r['tests']) for r in records] if records else TASKS
    references = {r['id']: r['reference'] for r in records} if records else REFERENCES
    families = {r['id']: r['family'] for r in records} if records else {}
    torch.set_num_threads(4)
    backend = NativeBackend(args.checkpoint, device='cuda', attention='flash')
    summary = {'server': backend.source, 'mode': 'raw', 'image': IMAGE,
               'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'tasks_sha256': hashlib.sha256(json.dumps(tasks).encode()).hexdigest(), 'tasks': []}
    with tempfile.TemporaryDirectory(prefix='microgpt-raw-') as folder:
        base = Path(folder); work = base / 'work'; work.mkdir(mode=0o755)
        for name, request, tests in tasks:
            function = ast.parse(references[name]).body[0]
            prefix = f'def {function.name}({ast.unparse(function.args)}):\n    """{request}"""\n'
            result = backend.complete(backend.tokenizer.encode(prefix), 768, 0, ['\ndef ', '\nclass '])
            code = prefix + result.text
            (work / 'solution.py').write_text(code)
            grader = base / 'grader.py'
            grader.write_text("import sys\nsys.path.insert(0, '/work')\nimport solution as s\n" + tests + "print('PASS')\n")
            outcome = grade(docker_command(work, grader, 'microgpt-raw-' + uuid.uuid4().hex))
            row = {'id': name, 'passed': outcome['passed'], 'grade': outcome['output'],
                   'finish_reason': result.finish_reason, 'completion_tokens': result.completion_tokens}
            row['family'] = families.get(name, name)
            summary['tasks'].append(row)
            groups = {}
            for item in summary['tasks']:
                group = groups.setdefault(item['family'], {'passed': 0, 'total': 0})
                group['passed'] += int(item['passed'])
                group['total'] += 1
            summary['families'] = groups
            (args.out / f'{name}.py').write_text(code)
            (args.out / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
