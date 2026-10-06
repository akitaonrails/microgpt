"""Separate code-format/interface failures from sandbox correctness, without execution."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path


def analyze(results, tasks):
    definitions={r['id']:r for r in tasks}
    report={'counts':Counter(),'families':{},'tasks':[]}
    for result in results['tasks']:
        task=definitions[result['id']]
        expected=next(node for node in ast.parse(task['reference']).body if isinstance(node, ast.FunctionDef))
        row={'id':result['id'],'passed':result['passed'],'syntax_valid':False,
             'requested_function_present':False,'parameter_names_match':False}
        try:
            tree=ast.parse(task['generated'])
            row['syntax_valid']=True
            function=next((n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==expected.name),None)
            row['requested_function_present']=function is not None
            if function is not None:
                def parameters(f):
                    return ([a.arg for a in f.args.posonlyargs], [a.arg for a in f.args.args],
                            [a.arg for a in f.args.kwonlyargs], f.args.vararg.arg if f.args.vararg else None,
                            f.args.kwarg.arg if f.args.kwarg else None)
                row['parameter_names_match']=parameters(function)==parameters(expected)
        except SyntaxError:pass
        for key in ('passed','syntax_valid','requested_function_present','parameter_names_match'):
            report['counts'][key]+=int(row[key])
        report['counts']['total']+=1
        family=report['families'].setdefault(task['family'],{'passed':0,'total':0})
        family['passed']+=int(row['passed']);family['total']+=1
        report['tasks'].append(row)
    report['note']='AST diagnostics do not execute code or alter original pass/fail. Parameter-name matches do not prove matching defaults or behavior.'
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--tasks-file',type=Path,default=Path('evaluation/programming-v1/development.jsonl'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    tasks=[json.loads(line) for line in args.tasks_file.open()]
    results=json.loads(args.results.read_text())
    for task in tasks:
        path=args.results.parent/(task['id']+'.py')
        task['generated']=path.read_text() if path.exists() else ''
    report=analyze(results,tasks)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['counts']))


if __name__=='__main__':main()
