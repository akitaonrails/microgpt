"""One preregistered scratch-capacity experiment, with measured rejection gates.

Reuses our original 1B-token randomly initialized control and the exact corpus.
Only architecture changes in pretraining. Both foundations then get the same
short instruction adapter. No automatic deployment or cloud provisioning.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--hours',type=float,default=12)
    p.add_argument('--port',type=int,default=8023)
    args=p.parse_args()
    if args.hours <= 0: p.error('hours must be positive')
    args.out.mkdir(parents=True,exist_ok=False)
    control=Path('runs/coding-v1/foundation/last.pt')
    tasks=Path('evaluation/programming-v1/development.jsonl')
    config=Path('configs/capacity-239m.toml')
    state={'status':'running','pid':os.getpid(),'started':datetime.now(timezone.utc).isoformat(),'stages':[],
           'hypothesis':'Increasing transformer capacity at fixed corpus/token budget improves executable code.',
           'control':str(control),'control_sha256':hashlib.sha256(control.read_bytes()).hexdigest(),
           'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
           'tasks_sha256':hashlib.sha256(tasks.read_bytes()).hexdigest(),
           'data_manifest_sha256':hashlib.sha256(Path('data/large-v1/meta.json').read_bytes()).hexdigest(),
           'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
           'training_hours_limit':args.hours,'automatic_promotion':False,
           'acceptance_set_used':False,'stopping_rule':'Reject nonfinite validation or loss > matched-step control + 0.2; no expansion past 1B exposures.'}
    stopping=False; child=None
    def save():
        state['updated']=datetime.now(timezone.utc).isoformat()
        tmp=args.out/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(args.out/'status.json')
    def stop(*_):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None: child.terminate()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    def run(command,log,timeout=None):
        nonlocal child
        if stopping: raise InterruptedError('Stopped before next operation')
        with log.open('a') as output:
            child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT)
            try: code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.terminate()
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired: child.kill();child.wait()
                raise
        child=None
        if stopping: raise InterruptedError('Stopped')
        if code: raise RuntimeError(f'Command exited {code}; inspect {log}')
    def raw(checkpoint,destination):
        run([sys.executable,'scripts/eval_raw_programming.py','--checkpoint',str(checkpoint),
             '--tasks-file',str(tasks),'--out',str(destination)],destination.with_suffix('.log'),1800)
        result=json.loads((destination/'results.json').read_text())
        return sum(r['passed'] for r in result['tasks'])
    def chat(checkpoint,destination):
        with destination.with_suffix('.server.log').open('w') as log:
            server=subprocess.Popen([sys.executable,'-m','microgpt.serve','--checkpoint',str(checkpoint),
                '--device','cuda','--attention','flash','--constrain-tools','--port',str(args.port)],stdout=log,stderr=subprocess.STDOUT)
            try:
                url=f'http://127.0.0.1:{args.port}'
                for _ in range(60):
                    if stopping: raise InterruptedError('Stopped')
                    if server.poll() is not None: raise RuntimeError('Server exited during startup')
                    try:
                        with urllib.request.urlopen(url+'/health',timeout=2) as response: health=json.load(response)
                        if health['source']['checkpoint'] != str(checkpoint): raise RuntimeError('Port belongs to another checkpoint')
                        break
                    except OSError: time.sleep(1)
                else: raise RuntimeError('Server did not start')
                run([sys.executable,'scripts/eval_programming.py','--mode','direct','--tasks-file',str(tasks),
                     '--limit','100','--url',url+'/v1','--out',str(destination)],destination.with_suffix('.log'),3600)
                result=json.loads((destination/'results.json').read_text())
                return sum(r['passed'] for r in result['tasks'])
            finally:
                server.terminate()
                try:server.wait(timeout=20)
                except subprocess.TimeoutExpired:server.kill();server.wait()
    save();training_seconds=0
    try:
        state['phase']='control_raw';save()
        state['control_raw']=raw(control,args.out/'control-raw');save()
        # Independent schedule stops avoid changing the cosine horizon at each tranche.
        control_metrics=[json.loads(line) for line in Path('runs/coding-v1/foundation/metrics.jsonl').read_text().splitlines()]
        foundation=args.out/'foundation'
        for step in (30518,61036,122071):
            remaining=args.hours*3600-training_seconds
            if remaining<=0: state['status']='budget_exhausted';return
            stage={'step':step,'status':'training'};state['stages'].append(stage);state['phase']='pretraining';save()
            command=[sys.executable,'-m','microgpt.train','--config',str(config),'--data','data/large-v1',
                '--out',str(foundation),'--device','cuda','--attention','flash','--sequence-length','2048',
                '--stop-at-step',str(step),'--max-seconds',str(remaining)]
            if (foundation/'last.pt').exists():command+=['--resume',str(foundation/'last.pt')]
            started=time.monotonic();run(command,args.out/'pretraining.log');training_seconds+=time.monotonic()-started
            metrics=json.loads((foundation/'metrics.jsonl').read_text().splitlines()[-1]);stage['metrics']=metrics
            state['training_seconds']=training_seconds
            if metrics['step']!=step: state['status']='budget_exhausted';return
            frozen=foundation/f'checkpoint-{step}.pt';os.link(foundation/'last.pt',frozen)
            stage['checkpoint']=str(frozen)
            comparison=min(control_metrics,key=lambda row:abs(row['step']-step))
            stage['control_metric']=comparison
            if not math.isfinite(metrics['val_loss']) or metrics['val_loss']>comparison['val_loss']+0.2:
                stage['status']='rejected_loss';state['status']='rejected_capacity_pilot';return
            stage['status']='raw_evaluation';save()
            stage['raw_passed']=raw(frozen,args.out/f'raw-{step}')
            stage['status']='evaluated';save()
        # A matched short adapter measures instruction transfer; it is not a full agent stage.
        state['phase']='matched_instruction_adapters';save()
        scores={}
        for label,parent in [('control',control),('candidate',foundation/'last.pt')]:
            remaining=args.hours*3600-training_seconds
            if remaining<=0:state['status']='budget_exhausted';return
            target=args.out/(label+'-adapter')
            command=[sys.executable,'-m','microgpt.finetune','--checkpoint',str(parent),
                '--data','data/programming-sft-v2','--out',str(target),'--steps','5000','--accumulation','2',
                '--lr','0.00005','--eval-every','500','--eval-examples','80','--device','cuda','--attention','flash',
                '--max-length','8192','--sources','opencodeinstruct','opencoder-educational','code-feedback','magicoder',
                '--balanced-sources','--max-seconds',str(remaining)]
            started=time.monotonic();run(command,args.out/(label+'-adapter.log'));training_seconds+=time.monotonic()-started
            metric=json.loads((target/'metrics.jsonl').read_text().splitlines()[-1])
            if metric['step']!=5000:state['status']='budget_exhausted';return
            scores[label]=chat(target/'last.pt',args.out/(label+'-direct'))
            state['adapter_scores']=scores;state['training_seconds']=training_seconds;save()
        state['gain_percentage_points']=scores['candidate']-scores['control']
        state['direct_gate_passed']=scores['candidate']>=80
        state['status']='ready_for_tool_curriculum' if state['direct_gate_passed'] else ('capacity_gain_needs_review' if state['gain_percentage_points']>=10 else 'no_meaningful_capacity_gain')
    except Exception as error:
        state['status']='interrupted' if stopping else 'failed';state['error']=str(error)
        if not stopping:raise
    finally:save()


if __name__ == '__main__':main()
