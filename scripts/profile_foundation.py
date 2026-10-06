"""Measure real 2K pretraining steps before budgeting a scratch experiment."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from microgpt.data import MixtureStream
from microgpt.inference import attention_context
from microgpt.model import GPT, ModelConfig
from microgpt.train import precision


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/large-v1'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=20)
    args=p.parse_args()
    if args.out.exists() or args.steps < 5: p.error('Use fresh output and at least five measured steps')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    meta=json.loads((args.data/'meta.json').read_text())
    stream=MixtureStream(args.data,'train',2048,meta['sources'])
    rows=[]
    for layers,width,heads in [(8,384,6),(12,768,12)]:
        torch.manual_seed(42)
        generator=torch.Generator().manual_seed(42)
        model=GPT(ModelConfig(vocab_size=meta['vocab_size'],context=32768,layers=layers,width=width,heads=heads)).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),weight_decay=0.1)
        torch.cuda.reset_peak_memory_stats()
        def step():
            optimizer.zero_grad(set_to_none=True)
            with precision(torch.device('cuda')), attention_context('flash'):
                for _ in range(4):
                    x,y=stream.batch(1,generator,'cuda')
                    _,loss=model(x,y)
                    (loss/4).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
        for _ in range(3): step()
        torch.cuda.synchronize(); start=time.monotonic()
        for _ in range(args.steps): step()
        torch.cuda.synchronize(); elapsed=time.monotonic()-start
        row={'layers':layers,'width':width,'parameters':sum(p.numel() for p in model.parameters()),
             'tokens_per_second':args.steps*8192/elapsed,'seconds':elapsed,
             'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9,
             'device':torch.cuda.get_device_name(),'steps':args.steps,'sequence_length':2048,
             'note':'Warmup excluded; small sampled workload, not sustained-run estimate.'}
        rows.append(row); print(json.dumps(row),flush=True)
        args.out.write_text(json.dumps(rows,indent=2)+'\n')
        del optimizer,model; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__': main()
