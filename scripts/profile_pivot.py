"""Measure the proposed model on real corpus windows; do not retain profile weights."""
import argparse
import json
from pathlib import Path
import time
import tomllib

import torch
from microgpt.data import MixtureStream
from microgpt.inference import attention_context
from microgpt.model import GPT, ModelConfig
from microgpt.train import precision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    config = tomllib.loads(args.config.read_text())
    meta = json.loads((args.data / 'meta.json').read_text())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(20261007)
    stream = MixtureStream(args.data, 'train', 2048, meta['sources'])
    generator = torch.Generator().manual_seed(20261007)
    model = GPT(ModelConfig(vocab_size=meta['vocab_size'], **config['model'])).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95), weight_decay=.1)
    settings = config['train']
    torch.cuda.reset_peak_memory_stats()
    losses = []
    def step():
        opt.zero_grad(set_to_none=True)
        value = 0.
        for _ in range(settings['accumulation']):
            x, y = stream.batch(settings['batch_size'], generator, 'cuda')
            with precision(torch.device('cuda')), attention_context('flash'):
                _, loss = model(x, y)
            if not torch.isfinite(loss): raise ValueError('Nonfinite profile loss')
            value += loss.item() / settings['accumulation']
            (loss / settings['accumulation']).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        return value
    for _ in range(3): step()
    torch.cuda.synchronize(); start = time.monotonic()
    for _ in range(30): losses.append(step())
    torch.cuda.synchronize(); elapsed = time.monotonic() - start
    result = {'parameters': sum(p.numel() for p in model.parameters()),
              'transformer_parameters': sum(p.numel() for p in model.parameters()) - model.embedding.weight.numel(),
              'tokens_per_second': 30 * 2048 * settings['batch_size'] * settings['accumulation'] / elapsed,
              'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
              'seconds': elapsed, 'losses': losses, 'attention': 'flash',
              'note': '30 real-data updates; excludes warmup, saving and evaluation; not a sustained-run guarantee.'}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
