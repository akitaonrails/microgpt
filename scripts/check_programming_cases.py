"""Execute benchmark reference/negative controls only in a restricted container."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import uuid
from eval_opencode import IMAGE

WORKER = '''import contextlib, json, os, types
for line in open('/cases.jsonl'):
    row = json.loads(line)
    result = {'id': row['id']}
    for label, code in [('reference', row['reference']), ('empty', '')]:
        s = types.ModuleType('solution')
        try:
            with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                exec(code, s.__dict__)
                exec(row['tests'], {'s': s})
            result[label] = True
        except Exception as error:
            result[label] = False
            result[label + '_error'] = str(error)
    print(json.dumps(result), flush=True)
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks-file',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        base=Path(directory);base.chmod(0o755)
        worker=base/'worker.py';worker.write_text(WORKER);worker.chmod(0o644)
        name='microgpt-cases-'+uuid.uuid4().hex
        command=['docker','run','--rm','--name',name,'--network','none','--read-only','--cap-drop','ALL',
            '--security-opt','no-new-privileges','--pids-limit','64','--memory','256m','--cpus','1',
            '--user','65534:65534','--tmpfs','/tmp:rw,noexec,size=16m',
            '--mount',f'type=bind,src={args.tasks_file.resolve()},dst=/cases.jsonl,readonly',
            '--mount',f'type=bind,src={worker},dst=/worker.py,readonly',IMAGE,'python','-I','-B','/worker.py']
        try: result=subprocess.run(command,capture_output=True,text=True,timeout=60,check=True)
        finally: subprocess.run(['docker','rm','-f',name],capture_output=True)
    rows=[json.loads(line) for line in result.stdout.splitlines()]
    args.out.write_text(json.dumps({'image':IMAGE,'tasks':rows},indent=2)+'\n')
    expected=sum(1 for _ in args.tasks_file.open())
    if len(rows)!=expected or not all(r['reference'] and not r['empty'] for r in rows):
        raise RuntimeError(f'Invalid controls; see {args.out}')
    print(f'{len(rows)} references passed and empty implementations rejected')


if __name__ == '__main__': main()
