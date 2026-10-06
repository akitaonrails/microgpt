"""Check one renamed variant per public reference against upstream tests in Docker.

Variants differ in identifiers; this validates a representative transformation per
source, not every generated identifier or full problem-spec correctness.
"""
import ast
import json
from pathlib import Path
import subprocess
import tempfile
import uuid
from microgpt.curate import rows
from prepare_programming_tools import WORKER
from prepare_interface_curriculum import rename
from eval_opencode import IMAGE


def main():
    selected={}
    for line in Path('data/programming-interface-v1.jsonl').open():
        record=json.loads(line)
        identifier=record['upstream_id']
        # The second occurrence is the first renamed variant.
        if identifier in selected and selected[identifier] is not None: continue
        if identifier not in selected: selected[identifier]=None;continue
        selected[identifier]=record
    checked=[]
    source=Path('/mnt/gigachad/microgpt/dataset/raw/opencoder-educational/educational_instruct/train-00000-of-00001.parquet')
    for row in rows(source):
        record=selected.get(row['seq_id'])
        if record is None:continue
        code=record['messages'][-1]['content'].removeprefix('```python\n').removesuffix('```')
        old=next(n.name for n in ast.parse(row['code']).body if isinstance(n,ast.FunctionDef))
        new=next(n.name for n in ast.parse(code).body if isinstance(n,ast.FunctionDef))
        tests=rename(ast.parse('\n'.join(row['testcase'])),old,new)
        checked.append({'seq_id':row['seq_id'],'code':code,'testcase':[tests]})
    with tempfile.TemporaryDirectory(prefix='microgpt-interface-check-') as directory:
        base=Path(directory);base.chmod(0o755)
        source=base/'input.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in checked));source.chmod(0o644)
        worker=base/'worker.py';worker.write_text(WORKER);worker.chmod(0o644)
        name='microgpt-interface-'+uuid.uuid4().hex
        command=['docker','run','--rm','--name',name,'--network','none','--read-only','--cap-drop','ALL',
            '--security-opt','no-new-privileges','--pids-limit','64','--memory','512m','--cpus','2','--user','65534:65534',
            '--tmpfs','/tmp:rw,noexec,size=16m','--mount',f'type=bind,src={source},dst=/input.jsonl,readonly',
            '--mount',f'type=bind,src={worker},dst=/worker.py,readonly',IMAGE,'python','-I','-B','/worker.py']
        try: result=subprocess.run(command,capture_output=True,text=True,check=True,timeout=7200)
        finally:subprocess.run(['docker','rm','-f',name],capture_output=True)
    results=[json.loads(line) for line in result.stdout.splitlines()]
    summary={'image':IMAGE,'selected':len(checked),'results':results,'passed':sum(r['passed'] for r in results),
        'note':'One renamed variant per upstream id; original upstream tests, not independent specification verification.'}
    Path('runs/interface-reference-checks.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}))
    if len(results)!=len(checked) or not all(r['passed'] for r in results):raise RuntimeError('Reference transformation failed; inspect result IDs')


if __name__=='__main__':main()
