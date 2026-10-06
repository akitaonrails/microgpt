"""Full-weight assistant-only training of OUR pretrained-from-scratch GPT."""
import argparse
from dataclasses import asdict
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import torch

from .model import GPT, ModelConfig
from .tokenizer import Tokenizer
from .protocol import digest, tokenizer_identity, VERSION
from .train import precision
from .inference import attention_context
from .pacing import GpuPacer
from .episodes import Episodes


def load_data(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('protocol') != VERSION:
        raise ValueError('Prepare scratch-protocol data with microgpt.trajectories')
    splits = {}
    for name in ('train', 'val'):
        splits[name] = Episodes(directory / f'{name}.jsonl', manifest['splits'][name]['sha256'])
    if splits['train'].groups & splits['val'].groups:
        raise ValueError('Training and validation groups overlap')
    return manifest, splits


def loss_for(model, example, device):
    ids = torch.tensor([example['input_ids'][:-1]], device=device)
    labels = torch.tensor(example['labels'][1:], device=device)
    positions = torch.where(labels != -100)[0]
    if not len(positions):
        raise ValueError('No assistant targets')
    _, loss = model(ids, targets=labels[positions][None], positions=positions, loss_chunk=128)
    return loss, len(positions)


def loss_for_batch(model, examples, device):
    """Right-pad independent sequences; preserve the mean of episode losses."""
    if not examples:
        raise ValueError('Empty microbatch')
    if len(examples) == 1:
        return loss_for(model, examples[0], device)
    selected = [torch.tensor([i for i, label in enumerate(item['labels'][1:]) if label != -100])
                for item in examples]
    if any(not len(positions) for positions in selected):
        raise ValueError('No assistant targets')
    ids = torch.zeros((len(examples), max(len(item['input_ids']) - 1 for item in examples)), dtype=torch.long)
    positions = torch.zeros((len(examples), max(map(len, selected))), dtype=torch.long)
    labels = torch.full_like(positions, -100)
    for row, (item, indices) in enumerate(zip(examples, selected)):
        ids[row, :len(item['input_ids']) - 1] = torch.tensor(item['input_ids'][:-1])
        positions[row, :len(indices)] = indices
        labels[row, :len(indices)] = torch.tensor(item['labels'][1:])[indices]
    _, loss = model(ids.to(device), targets=labels.to(device), positions=positions.to(device),
                    loss_chunk=128, loss_reduction='example_mean')
    return loss, sum(map(len, selected))


@torch.no_grad()
def evaluate(model, examples, device, attention, by_source=False, pacer=None):
    model.eval()
    total = tokens = 0
    sources = {}
    with precision(device), attention_context(attention):
        for example in examples:
            loss, count = loss_for(model, example, device)
            total += loss.item() * count
            tokens += count
            source = example.get('source', 'unknown')
            subtotal, subcount = sources.get(source, (0, 0))
            sources[source] = (subtotal + loss.item() * count, subcount + count)
            if pacer is not None:
                pacer.wait()
    model.train()
    overall = total / tokens
    return (overall, {key: value / count for key, (value, count) in sources.items()}) if by_source else overall


def bucket_weights(buckets, weights):
    """Keep a source's requested probability constant across its length bins."""
    from collections import Counter
    counts = Counter(key.split('@')[0] for key in buckets)
    if set(counts) != set(weights):
        raise ValueError(f'Source weights must cover retained sources exactly: {sorted(counts)}')
    return [weights[key.split('@')[0]] / counts[key.split('@')[0]] for key in buckets]


def validation_sample(buckets, count, seed=None):
    """Round-robin sources, optionally shuffling within each without changing training."""
    ordered = [list(buckets[source]) for source in sorted(buckets)]
    if seed is not None:
        rng = random.Random(seed)
        for bucket in ordered:
            rng.shuffle(bucket)
    return [index for row in zip_longest(*ordered) for index in row if index is not None][:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument('--checkpoint', type=Path, help='Our pretraining checkpoint: start SFT with a new optimizer')
    start.add_argument('--resume', type=Path, help='Our SFT checkpoint: restore optimizer and RNG')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--microbatch', type=int, default=1, help='Independent padded episodes per forward pass; effective batch is microbatch × accumulation')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--eval-examples', type=int, default=30)
    parser.add_argument('--validation-seed', type=int, help='Fixed random sample within validation sources; omitted preserves legacy source order')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--attention', choices=['flash', 'sdpa', 'math'], default='flash')
    parser.add_argument('--max-length', type=int, default=32768, help='Curriculum stage: exclude whole episodes exceeding this length')
    parser.add_argument('--min-tool-calls', type=int, default=0, help='Select complete longer-horizon episodes for a dedicated stage')
    parser.add_argument('--source-weight', action='append', default=[], metavar='SOURCE:WEIGHT',
                        help='Explicit source sampling weights; repeats allowed for different sources')
    parser.add_argument('--balanced-sources', action='store_true', help='Sample a source uniformly, then an episode within it')
    parser.add_argument('--balanced-lengths', action='store_true', help='Balance source/8K/16K/32K bins so long histories receive regular updates')
    parser.add_argument('--short-replay', type=float, default=0, help='Fraction of updates sampled from the general <=8K curriculum to reduce forgetting')
    parser.add_argument('--sources', nargs='+', help='Restrict the main curriculum to these source names; replay still uses all sources')
    parser.add_argument('--max-seconds', type=float, help='Save and stop after this wall-clock training budget')
    parser.add_argument('--allow-attention-change', action='store_true',
                        help='Explicitly resume optimizer/RNG with a different attention backend; record the change in lineage')
    args = parser.parse_args()
    if min(args.steps, args.accumulation, args.microbatch, args.eval_every, args.eval_examples) < 1 or args.lr <= 0:
        parser.error('Training settings must be positive')
    if not 0 <= args.short_replay < 1:
        parser.error('short-replay must be in [0, 1)')
    if args.attention == 'flash' and args.device != 'cuda':
        parser.error('Flash Attention requires CUDA')
    if args.out.exists() and not args.resume:
        parser.error('Use a fresh output directory or resume')
    source_weights = {}
    for item in args.source_weight:
        name, value = item.rsplit(':', 1)
        weight = float(value)
        if name in source_weights or not math.isfinite(weight) or weight <= 0:
            parser.error('Source weights must be unique, finite and positive')
        source_weights[name] = weight
    manifest, data = load_data(args.data)
    path = args.resume or args.checkpoint
    saved = torch.load(path, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.load(path.parent)
    if tokenizer_identity(tokenizer) != manifest['tokenizer_fingerprint']:
        raise ValueError('SFT and checkpoint tokenizers differ; never replace learned token IDs')
    config = ModelConfig(**saved['model_config'])
    if config.vocab_size != tokenizer.vocab_size or manifest['max_length'] > config.context:
        raise ValueError('Vocabulary/context mismatch; prepare SFT within the checkpoint context')
    settings = {'stage': 'sft', 'dataset': digest(manifest), 'lr': args.lr,
                'accumulation': args.accumulation, 'microbatch': args.microbatch, 'seed': args.seed,
                'eval_examples': args.eval_examples, 'validation_seed': args.validation_seed,
                'attention': args.attention, 'device': args.device,
                'max_length': args.max_length, 'min_tool_calls': args.min_tool_calls,
                'balanced_sources': args.balanced_sources, 'balanced_lengths': args.balanced_lengths,
                'short_replay': args.short_replay, 'sources': sorted(args.sources) if args.sources else None,
                'source_weights': source_weights}
    previous_settings = dict(saved['settings'])
    previous_settings.setdefault('microbatch', 1)
    previous_settings.setdefault('validation_seed', None)
    for key, default in [('max_length', 32768), ('min_tool_calls', 0), ('balanced_sources', False), ('balanced_lengths', False), ('short_replay', 0), ('sources', None), ('source_weights', {})]:
        previous_settings.setdefault(key, default)
    backend_changed = bool(args.resume and args.allow_attention_change and previous_settings['attention'] != settings['attention'])
    if backend_changed:
        previous_settings['attention'] = settings['attention']
    if args.resume and previous_settings != settings:
        raise ValueError('Resume requires identical SFT data/settings')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = GPT(config).to(device)
    model.load_state_dict(saved['model'])
    model.gradient_checkpointing = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = random.Random(args.seed)
    step = 0
    tokens_seen = saved.get('assistant_tokens_seen', 0) if args.resume else 0
    best_loss = saved.get('best_val_loss', float('inf')) if args.resume else float('inf')
    origin = saved.get('origin', {'initialization': 'random'})
    if backend_changed:
        origin.setdefault('attention_changes', []).append({'from': saved['settings']['attention'],
            'to': settings['attention'], 'step': saved['step'], 'reason': 'Explicit resume backend override'})
    if args.checkpoint:
        parent_stage = saved.get('settings', {}).get('stage', 'pretraining')
        origin = {**origin, 'parent_checkpoint': str(path), 'parent_step': saved['step'], 'parent_stage': parent_stage}
        if parent_stage != 'sft':
            origin.update(pretraining_checkpoint=str(path), pretraining_step=saved['step'])
    if args.resume:
        optimizer.load_state_dict(saved['optimizer'])
        rng.setstate(saved['example_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        step = saved['step']
    if step >= args.steps:
        parser.error('--steps must exceed resumed step')
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.out)
    (args.out / 'meta.json').write_text(json.dumps({'tokenizer_fingerprint': tokenizer_identity(tokenizer), 'protocol': VERSION}))
    train_buckets = data['train'].buckets(args.max_length, args.min_tool_calls, args.balanced_lengths)
    val_buckets = data['val'].buckets(args.max_length, args.min_tool_calls, args.balanced_lengths)
    if args.sources:
        train_buckets = {k: v for k, v in train_buckets.items() if k.split('@')[0] in args.sources}
        val_buckets = {k: v for k, v in val_buckets.items() if k.split('@')[0] in args.sources}
    if not train_buckets or not val_buckets:
        raise ValueError('Curriculum requires retained training and held-out episodes')
    validation_indices = validation_sample(val_buckets, args.eval_examples, args.validation_seed)
    validation = [data['val'][i] for i in validation_indices]
    training = list(train_buckets.values()) if source_weights or args.balanced_sources or args.balanced_lengths else [[i for bucket in train_buckets.values() for i in bucket]]
    sampling_weights = bucket_weights(train_buckets, source_weights) if source_weights else None
    replay = list(data['train'].buckets(min(8192, args.max_length)).values())
    if args.short_replay and not replay:
        raise ValueError('Short replay requires retained <=8K episodes')
    print(json.dumps({'training_by_source': {k: len(v) for k, v in train_buckets.items()},
                      'validation_by_source': {k: len(v) for k, v in val_buckets.items()},
                      'max_length': args.max_length, 'min_tool_calls': args.min_tool_calls, 'short_replay': args.short_replay}), flush=True)
    pacer = GpuPacer(device)
    print(json.dumps({'gpu_duty_cycle': pacer.fraction}), flush=True)
    print(json.dumps({'step': step, 'val_loss': evaluate(model, validation, device, args.attention, pacer=pacer),
                      'parameters': sum(p.numel() for p in model.parameters())}), flush=True)
    started = time.monotonic()
    stopping = []
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopping.append(True))
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    while step < args.steps:
        optimizer.zero_grad(set_to_none=True)
        total = 0
        with precision(device), attention_context(args.attention):
            for _ in range(args.accumulation):
                examples = []
                for _ in range(args.microbatch):
                    buckets = replay if args.short_replay and rng.random() < args.short_replay else training
                    bucket = rng.choices(training, weights=sampling_weights, k=1)[0] if buckets is training and sampling_weights else rng.choice(buckets)
                    examples.append(data['train'][rng.choice(bucket)])
                loss, count = loss_for_batch(model, examples, device)
                tokens_seen += count
                total += loss.item() / args.accumulation
                (loss / args.accumulation).backward()
                pacer.wait()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        pacer.wait()
        step += 1
        budget_reached = bool(stopping) or (args.max_seconds is not None and time.monotonic() - started >= args.max_seconds)
        if step % args.eval_every and step != args.steps and not budget_reached:
            continue
        val_loss, val_sources = evaluate(model, validation, device, args.attention, by_source=True, pacer=pacer)
        metrics = {'step': step, 'train_loss': total, 'val_loss': val_loss, 'val_loss_by_source': val_sources,
                   'assistant_tokens_seen': tokens_seen, 'gpu_duty_cycle': pacer.fraction,
                   'seconds': round(time.monotonic() - started, 1), 'budget_reached': budget_reached,
                   'peak_memory_gb': torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0}
        improved = metrics['val_loss'] < best_loss
        best_loss = min(best_loss, metrics['val_loss'])
        payload = {'model_config': asdict(config), 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'assistant_tokens_seen': tokens_seen, 'best_val_loss': best_loss,
                   'step': step, 'settings': settings, 'data': manifest, 'example_rng': rng.getstate(),
                   'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                   'origin': origin, 'protocol': VERSION}
        torch.save(payload, args.out / 'last.tmp')
        (args.out / 'last.tmp').replace(args.out / 'last.pt')
        if improved:
            # Link the immutable saved file; the next last.pt is an atomic replacement.
            temporary = args.out / 'best.tmp'
            temporary.unlink(missing_ok=True)
            os.link(args.out / 'last.pt', temporary)
            temporary.replace(args.out / 'best.pt')
        with (args.out / 'metrics.jsonl').open('a') as output:
            output.write(json.dumps(metrics) + '\n')
        print(json.dumps(metrics), flush=True)
        if budget_reached:
            break


if __name__ == '__main__':
    main()
