"""Measure short-episode SFT microbatches at a fixed effective batch and data order.

Runs real forward/backward/AdamW updates on disposable copies of our checkpoint.
No weights are saved or served. Run when other training/evaluation is idle.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import subprocess
import time

import torch

from microgpt.finetune import load_data, loss_for_batch, bucket_weights
from microgpt.inference import attention_context
from microgpt.model import GPT, ModelConfig
from microgpt.protocol import tokenizer_identity
from microgpt.tokenizer import Tokenizer
from microgpt.train import precision
from microgpt.pacing import GpuPacer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--effective-batch', type=int, default=32)
    parser.add_argument('--microbatches', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--attention', choices=['math', 'flash', 'sdpa'], default='math')
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--source-weight', action='append', default=[])
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a fresh output')
    if min(args.effective_batch, args.steps, *args.microbatches) < 1 or args.warmup < 0:
        parser.error('Invalid batch/step counts')
    if any(args.effective_batch % size for size in args.microbatches):
        parser.error('Each microbatch must divide the effective batch')
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if saved.get('origin', {}).get('initialization') != 'random':
        raise ValueError('Expected our scratch checkpoint')
    manifest, data = load_data(args.data)
    if tokenizer_identity(Tokenizer.load(args.checkpoint.parent)) != manifest['tokenizer_fingerprint']:
        raise ValueError('Checkpoint and data tokenizers differ')
    rng = random.Random(42)
    buckets = data['train'].buckets(args.max_length)
    weights = {name: float(value) for name, value in (item.rsplit(':', 1) for item in args.source_weight)}
    sampling_weights = bucket_weights(buckets, weights) if weights else None
    eligible = list(buckets.values()) if weights else [[index for bucket in buckets.values() for index in bucket]]
    def example():
        bucket = rng.choices(eligible, weights=sampling_weights, k=1)[0] if weights else rng.choice(eligible)
        return data['train'][rng.choice(bucket)]
    batches = [[example() for _ in range(args.effective_batch)]
               for _ in range(args.steps + args.warmup)]
    report = {'checkpoint': str(args.checkpoint), 'data_manifest': manifest,
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'effective_batch': args.effective_batch, 'attention': args.attention,
              'max_length': args.max_length, 'source_weights': weights,
              'retained_by_source': {key: len(value) for key, value in buckets.items()},
              'steps': args.steps, 'warmup': args.warmup, 'results': [],
              'note': 'Same examples and episode-mean objective; floating-point execution can differ. Data preloaded for profiling; real training also pays random-access I/O.'}
    device = torch.device('cuda')
    for size in args.microbatches:
        model = GPT(ModelConfig(**saved['model_config'])).to(device).train()
        model.load_state_dict(saved['model'])
        model.gradient_checkpointing = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
        pacer = GpuPacer(device)
        elapsed = 0
        tokens = 0
        result = {'microbatch': size, 'accumulation': args.effective_batch // size,
                  'gpu_duty_cycle': pacer.fraction}
        try:
            for step, examples in enumerate(batches):
                if step == args.warmup:
                    torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                count = 0
                for start in range(0, len(examples), size):
                    with precision(device), attention_context(args.attention):
                        loss, targets = loss_for_batch(model, examples[start:start + size], device)
                        (loss / result['accumulation']).backward()
                    pacer.wait()
                    count += targets
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                pacer.wait()
                torch.cuda.synchronize()
                if step >= args.warmup:
                    elapsed += time.perf_counter() - started
                    tokens += count
            result.update(seconds=elapsed, assistant_tokens=tokens, target_tokens_per_second=tokens / elapsed,
                          peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9)
        except torch.cuda.OutOfMemoryError:
            result['error'] = 'CUDA out of memory; no training weights saved'
        finally:
            loss = None
            optimizer.zero_grad(set_to_none=True)
            del optimizer, model
            gc.collect()
            torch.cuda.empty_cache()
        report['results'].append(result)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
