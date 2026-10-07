"""Freeze a small public-code memorization diagnostic, never an acceptance set."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

from microgpt.tokenizer import Tokenizer
from prepare_interface_curriculum import rename


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.source / 'manifest.json').read_text())
    assert sha(args.source / 'code.jsonl') == manifest['code_sha256']
    tokenizer = Tokenizer.load(args.tokenizer)
    selected = {'train': [], 'val': []}
    groups = set()
    for line in (args.source / 'code.jsonl').open():
        r = json.loads(line)
        split = 'val' if int(hashlib.sha256(r['group'].encode()).hexdigest()[:8], 16) % 20 == 0 else 'train'
        code = r['messages'][-1]['content'].removeprefix('```python\n').removesuffix('```')
        tree = ast.parse(code)
        functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))]
        # Bounded, plain functions make a failed memorization result interpretable.
        if len(functions) != 1 or not isinstance(functions[0], ast.FunctionDef) or functions[0].decorator_list:
            continue
        f = functions[0]
        if len(tokenizer.encode(code)) > 160 or r['group'] in groups or len(selected[split]) == 16:
            continue
        if any(isinstance(n, ast.Constant) and isinstance(n.value, str) and f.name in n.value for n in ast.walk(tree)):
            continue
        if any(isinstance(n, ast.Name) and n.id == f.name and isinstance(n.ctx, ast.Store) for n in ast.walk(f)):
            continue
        selected[split].append(r)
        groups.add(r['group'])
        if all(len(v) == 16 for v in selected.values()):
            break
    assert all(len(v) == 16 for v in selected.values())
    needed = {r['upstream_id'] for items in selected.values() for r in items}
    references = {}
    for line in (args.source / 'references.jsonl').open():
        r = json.loads(line)
        if r['seq_id'] in needed:
            references[r['seq_id']] = r
    assert set(references) == needed
    tasks = []
    with (args.out / 'curriculum.jsonl').open('x') as out:
        for split, items in selected.items():
            for index, record in enumerate(items):
                out.write(json.dumps(record) + '\n')
                ref = references[record['upstream_id']]
                code = record['messages'][-1]['content'].removeprefix('```python\n').removesuffix('```')
                prompt = next(m['content'] for m in record['messages'] if m['role'] == 'user')
                tests = '\n'.join(ref['testcase'])
                family = 'seen_train' if split == 'train' else 'unseen_source'
                task = {'id': f'{family}_{index}', 'family': family, 'upstream_id': record['upstream_id'],
                        'prompt': prompt, 'reference': code, 'tests': 'globals().update(vars(s))\n' + tests + '\n'}
                tasks.append(task)
                if split == 'train':
                    tree = ast.parse(code)
                    old = next(n.name for n in tree.body if isinstance(n, ast.FunctionDef))
                    new = f'probe_operation_{index}'
                    tasks.append({**task, 'id': f'renamed_{index}', 'family': 'renamed',
                        'prompt': re.sub(r'\b' + re.escape(old) + r'\b', new, prompt),
                        'reference': rename(tree, old, new),
                        'tests': 'globals().update(vars(s))\n' + rename(ast.parse(tests), old, new)})
    with (args.out / 'tasks.jsonl').open('x') as out:
        for task in tasks:
            out.write(json.dumps(task) + '\n')
    (args.out / 'provenance.json').write_text(json.dumps({
        'source_manifest': manifest, 'curriculum_sha256': sha(args.out / 'curriculum.jsonl'),
        'tasks_sha256': sha(args.out / 'tasks.jsonl'), 'counts': {'train': 16, 'val': 16, 'renamed': 16},
        'selection': 'First unique original source groups with one plain function, <=160 code tokens; original hash split retained.',
        'limitations': 'Seen and renamed tasks are memorization/interface diagnostics. Unseen sources are only held out from this SFT subset, not necessarily foundation pretraining. No acceptance tasks used.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
