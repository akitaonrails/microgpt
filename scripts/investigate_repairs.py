"""Audit repair supervision and teacher-forced held-out edit decisions.

Read-only diagnostics: no optimizer updates, generated-code execution or service
changes. Correct histories are supplied, so results do not establish agent success.
"""
import argparse
from collections import Counter
import copy
import difflib
import gc
import hashlib
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from microgpt.chat import ChatRequest, normalize
from microgpt.inference import NativeBackend, attention_context
from microgpt.pacing import GpuPacer
from microgpt.protocol import body, canonical_messages, prompt_ids, tokenizer_identity
from microgpt.tokenizer import Tokenizer
from microgpt.train import precision
from microgpt.trajectories import assistant_examples


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def edit_pair(record, tokenizer):
    messages, tools = normalize(ChatRequest(model='diagnostic', messages=record['messages'], tools=record['tools']))
    messages = canonical_messages(messages)
    index = next(i for i, m in enumerate(messages) if any(
        c['function']['name'] == 'edit' for c in m.get('tool_calls', [])))
    correct = messages[index]
    if len(correct['tool_calls']) != 1 or correct['tool_calls'][0]['function']['name'] != 'edit':
        raise ValueError('This probe requires a single edit call in the edit turn')
    noop = copy.deepcopy(correct)
    args = noop['tool_calls'][0]['function']['arguments']
    args['newString'] = args['oldString']
    wanted = tokenizer.encode(body(correct)) + [tokenizer.eos]
    copied = tokenizer.encode(body(noop)) + [tokenizer.eos]
    first = next(i for i, (a, b) in enumerate(zip(wanted, copied)) if a != b)
    changed = set()
    for tag, _, _, start, end in difflib.SequenceMatcher(a=copied, b=wanted, autojunk=False).get_opcodes():
        if tag != 'equal':
            changed.update(range(start, end))
            if start == end and start < len(wanted):
                changed.add(start)  # Next token decides whether to delete a copied span.
    prefix = prompt_ids(tokenizer, messages[:index], tools)
    return prefix, wanted, copied, first, sorted(changed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('data/public-code-multiseed-v1/repairs.jsonl'))
    p.add_argument('--tokenizer', type=Path, default=Path('data/verified-opencodeinstruct-sft-v1/tokenizer'))
    p.add_argument('--checkpoint', type=Path, nargs='*', default=[])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=64)
    args = p.parse_args()
    if args.out.exists() or args.count < 1:
        p.error('Use a fresh output and positive sample count')
    torch.set_num_threads(4)
    tokenizer = Tokenizer.load(args.tokenizer)
    records = [json.loads(line) for line in args.source.open()]
    totals = Counter(); patterns = Counter(); mutations = Counter(); heldout = []
    for record in records:
        mutations[record['mutation']] += 1
        pattern = []
        for message in record['messages']:
            if message['role'] == 'assistant':
                pattern += [call['function']['name'] for call in message.get('tool_calls', [])]
                totals['assistant_targets'] += len(tokenizer.encode(body(message))) + 1
        patterns[' -> '.join(pattern)] += 1
        prefix, wanted, copied, first, changed = edit_pair(record, tokenizer)
        totals['edit_targets'] += len(wanted)
        totals['changed_vs_noop_targets'] += len(changed)
        totals['records'] += 1
        if int(hashlib.sha256(str(record['group']).encode()).hexdigest()[:8], 16) % 20 == 0:
            heldout.append(record)
    random.Random(20261007).shuffle(heldout)
    selected = heldout[:args.count]
    report = {'source': str(args.source), 'source_sha256': sha(args.source),
              'counts': dict(totals), 'patterns': dict(patterns), 'mutations': dict(mutations),
              'changed_target_fraction': totals['changed_vs_noop_targets'] / totals['assistant_targets'],
              'heldout_pool': len(heldout), 'selected_ids': [r['upstream_id'] for r in selected],
              'seed': 20261007, 'checkpoints': [],
              'limits': 'Teacher-forced source-held-out diagnostic, not rollout accuracy. Changed tokens are BPE alignment differences versus a no-op; not a semantic token oracle. No benchmark solutions used.'}
    def save():
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    save()
    for checkpoint in args.checkpoint:
        backend = NativeBackend(checkpoint, device='cuda', attention='flash', prefix_cache=False)
        if tokenizer_identity(backend.tokenizer) != tokenizer_identity(tokenizer):
            raise ValueError('Probe and checkpoint tokenizers differ')
        pacer = GpuPacer(backend.device)
        result = {'checkpoint': str(checkpoint), 'sha256': sha(checkpoint), 'records': []}
        for record in selected:
            prefix, wanted, copied, first, changed = edit_pair(record, tokenizer)
            packed = assistant_examples(record, tokenizer, 8192, pack=True, canonical_tool_ids=True)[0]
            target_positions = [i for i, label in enumerate(packed['labels']) if label != -100]
            assert packed['input_ids'][:len(prefix)] == prefix
            positions = torch.tensor([i - 1 for i in target_positions], device=backend.device)
            labels = torch.tensor([packed['labels'][i] for i in target_positions], device=backend.device)
            inputs = torch.tensor([packed['input_ids'][:-1]], device=backend.device)
            with torch.inference_mode(), precision(backend.device), attention_context('flash'):
                logits, _ = backend.model(inputs, positions=positions)
                logits = logits[0].float()
                losses = F.cross_entropy(logits, labels, reduction='none')
                rows = [target_positions.index(len(prefix) + i) for i in changed]
                decision = target_positions.index(len(prefix) + first)
                delta = (logits[decision, wanted[first]] - logits[decision, copied[first]]).item()
                item = {'id': record['upstream_id'], 'mutation': record['mutation'],
                        'targets': len(target_positions), 'changed_targets': len(rows),
                        'all_nll_sum': losses.sum().item(), 'changed_nll_sum': losses[rows].sum().item(),
                        'changed_top1_correct': int((logits[rows].argmax(-1) == labels[rows]).sum().item()),
                        'correct_vs_noop_logit_margin': delta,
                        'first_difference_correct_top1': logits[decision].argmax().item() == wanted[first]}
            result['records'].append(item)
            pacer.wait()
        rows = result['records']
        result['summary'] = {'count': len(rows),
            'all_target_nll': sum(r['all_nll_sum'] for r in rows) / sum(r['targets'] for r in rows),
            'changed_target_nll': sum(r['changed_nll_sum'] for r in rows) / sum(r['changed_targets'] for r in rows),
            'changed_top1_accuracy': sum(r['changed_top1_correct'] for r in rows) / sum(r['changed_targets'] for r in rows),
            'correct_preferred_to_noop': sum(r['correct_vs_noop_logit_margin'] > 0 for r in rows),
            'first_difference_correct_top1': sum(r['first_difference_correct_top1'] for r in rows)}
        report['checkpoints'].append(result); save()
        print(json.dumps({'checkpoint': str(checkpoint), **result['summary']}), flush=True)
        del backend, logits, losses, inputs, positions, labels
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
