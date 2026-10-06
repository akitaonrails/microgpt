"""Teach public programming references under varied requested function names.

Uses only previously sandbox-checked source references. No benchmark imports.
Renaming is restricted to a single undecorated top-level function without nested
scopes; originals and renamed variants keep the source's split group. No code runs.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import re
from microgpt.programming_data import GENERAL_PROMPT
from microgpt.protocol import reasoning_instruction


def rename(tree, old, new):
    class Rename(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == old: node.name = new
            return self.generic_visit(node)
        def visit_Name(self, node):
            if node.id == old: node.id = new
            return node
    return ast.unparse(Rename().visit(copy.deepcopy(tree))) + '\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('data/educational-tools-v1.jsonl'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--variants',type=int,default=8)
    args=p.parse_args()
    if args.variants < 1: p.error('variants must be positive')
    stats={'source_records':0,'episodes':0,'skipped':0}
    rng=random.Random(20261006)
    with args.out.open('x') as out:
        for line in args.input.open():
            r=json.loads(line)
            edit=next(m['tool_calls'][0]['function']['arguments'] for m in r['messages']
                if m.get('tool_calls') and m['tool_calls'][0]['function']['name']=='edit')
            tree=ast.parse(edit['newString'])
            functions=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Lambda))]
            if len(functions)!=1 or not isinstance(functions[0],ast.FunctionDef) or functions[0].decorator_list:
                stats['skipped']+=1;continue
            f=functions[0]
            # Avoid changing a locally rebound name or reflective string lookup.
            if any(isinstance(n,ast.Name) and n.id==f.name and isinstance(n.ctx,ast.Store) for n in ast.walk(f)):
                stats['skipped']+=1;continue
            instruction=json.loads(next(m['content'] for m in r['messages'] if m['role']=='user'))
            instruction=instruction.removesuffix(' Implement this in solution.py and run check.')
            stats['source_records']+=1
            for index in range(args.variants):
                name=f.name if index==0 else rng.choice(['compute','handle','convert','process','evaluate'])+'_'+format(rng.getrandbits(24),'x')
                code=rename(tree,f.name,name)
                prompt=re.sub(r'\b'+re.escape(f.name)+r'\b',name,instruction)
                prompt+=f' Use the exact function signature {name}({ast.unparse(f.args)}). Return only Python code in a fenced block.'
                record={'group':r['group'],'source':'programming-interface','license':r['license'],
                    'upstream_id':r.get('upstream_id'),'construction':'AST-renamed previously checked public reference; variants not separately executed.',
                    'messages':[{'role':'system','content':reasoning_instruction('none')},
                        {'role':'system','content':GENERAL_PROMPT},{'role':'user','content':prompt},
                        {'role':'assistant','content':'```python\n'+code+'```'}], 'tools':[]}
                out.write(json.dumps(record)+'\n');stats['episodes']+=1
    stats['input_sha256']=hashlib.sha256(args.input.read_bytes()).hexdigest()
    stats['output_sha256']=hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix('.source.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats),flush=True)


if __name__=='__main__':main()
