"""Check one renamed variant per public reference against upstream tests in Docker.

Variants differ in identifiers; this validates a representative transformation per
source, not every generated identifier or full problem-spec correctness.
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
from prepare_programming_tools import WORKER
from prepare_interface_curriculum import rename
from eval_opencode import IMAGE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/programming-interface-v1.jsonl'))
    parser.add_argument('--source', type=Path, default=Path('/mnt/gigachad/microgpt/dataset/raw/opencoder-educational/educational_instruct/train-00000-of-00001.parquet'))
    parser.add_argument('--out', type=Path, default=Path('runs/interface-reference-checks.json'))
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a fresh output path to preserve previous evidence')
    selected={}
    for line in args.input.open():
        record=json.loads(line)
        identifier=record['upstream_id']
        # The second occurrence is the first renamed variant.
        if identifier in selected and selected[identifier] is not None: continue
        if identifier not in selected: selected[identifier]=None;continue
        selected[identifier]=record
    if not selected or any(record is None for record in selected.values()):
        raise ValueError('Expected an original and a renamed variant for every source ID')
    checked=[]
    preparation_failures=[]
    for row in rows(args.source):
        record=selected.get(row['seq_id'])
        if record is None:continue
        code=record['messages'][-1]['content'].removeprefix('```python\n').removesuffix('```')
        old=next(n.name for n in ast.parse(row['code']).body if isinstance(n,ast.FunctionDef))
        new=next(n.name for n in ast.parse(code).body if isinstance(n,ast.FunctionDef))
        try:
            tests=rename(ast.parse('\n'.join(normalize_reference_tests(row['testcase']))),old,new)
        except (SyntaxError, ValueError) as error:
            preparation_failures.append({'id':row['seq_id'], 'passed':False, 'error':str(error)})
            continue
        checked.append({'seq_id':row['seq_id'],'code':code,'testcase':[tests]})
    if len(checked) + len(preparation_failures) != len(selected):
        raise ValueError('Some curriculum source IDs are missing from the upstream shard')
    with tempfile.TemporaryDirectory(prefix='microgpt-interface-check-') as directory:
        base=Path(directory);base.chmod(0o755)
        source=base/'input.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in checked));source.chmod(0o644)
        worker=base/'worker.py';worker.write_text(WORKER);worker.chmod(0o644)
        name='microgpt-interface-'+uuid.uuid4().hex
        command=['docker','run','--rm','--name',name,'--network','none','--read-only','--cap-drop','ALL',
            '--security-opt','no-new-privileges','--pids-limit','64','--memory','512m','--cpus','2','--user','65534:65534',
            '--tmpfs','/tmp:rw,noexec,size=16m','--mount',f'type=bind,src={source},dst=/input.jsonl,readonly',
            '--mount',f'type=bind,src={worker},dst=/worker.py,readonly',IMAGE,'python','-I','-B','/worker.py']
        try: result=subprocess.run(command,capture_output=True,text=True,check=True,timeout=max(7200, len(checked)*2))
        finally:subprocess.run(['docker','rm','-f',name],capture_output=True)
    results=preparation_failures+[json.loads(line) for line in result.stdout.splitlines()]
    summary={'image':IMAGE,'selected':len(selected),'results':results,'passed':sum(r['passed'] for r in results),
        'note':'One renamed variant per upstream id; original upstream tests, not independent specification verification.'}
    for key, path in [('input_sha256', args.input), ('source_sha256', args.source)]:
        with path.open('rb') as stream:
            summary[key] = hashlib.file_digest(stream, 'sha256').hexdigest()
    args.out.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}))
    if len(results)!=len(selected) or not all(r['passed'] for r in results):raise RuntimeError('Reference transformation failed; inspect result IDs')


if __name__=='__main__':main()
