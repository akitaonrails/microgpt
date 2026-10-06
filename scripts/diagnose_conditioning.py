"""Compare reference-answer likelihood with and without the problem statement.

This diagnostic does not execute code, train weights, or grade programming ability.
It also verifies reconstructed preparation and math/Flash loss consistency.
"""
import argparse
import ast
import json
from pathlib import Path
import random

import torch

from microgpt.episodes import Episodes
from microgpt.finetune import loss_for
from microgpt.inference import NativeBackend, attention_context
from microgpt.protocol import digest
from microgpt.trajectories import assistant_examples
from microgpt.train import precision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--curriculum', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=80)
    parser.add_argument('--seed', type=int, default=20261007)
    args = parser.parse_args()
    if args.out.exists() or args.count < 1:
        parser.error('Use a fresh output and positive sample size')
    torch.set_num_threads(4)
    manifest = json.loads((args.data / 'manifest.json').read_text())
    examples = Episodes(args.data / 'val.jsonl', manifest['splits']['val']['sha256'])
    indices = random.Random(args.seed).sample(range(len(examples)), args.count)
    selected = {examples[index]['episode']: examples[index] for index in indices}
    records = {}
    for line in args.curriculum.open():
        record = json.loads(line)
        key = digest({'messages': record['messages'], 'tools': record.get('tools', [])})
        if key in selected:
            records[key] = record
    if len(records) != args.count:
        raise ValueError('Expected one complete single-answer episode per selected record')
    backend = NativeBackend(args.checkpoint, device='cuda', attention='flash', prefix_cache=False)
    report = {'checkpoint': args.checkpoint, 'seed': args.seed, 'manifest': manifest, 'records': []}
    with torch.inference_mode(), precision(backend.device):
        for key, record in records.items():
            example = selected[key]
            full = assistant_examples(record, backend.tokenizer, manifest['max_length'], pack=True)[0]
            if full['input_ids'] != example['input_ids'] or full['labels'] != example['labels']:
                raise ValueError('Reconstructed input/labels differ from prepared example')
            code = record['messages'][-1]['content'].removeprefix('```python\n').removesuffix('```')
            function = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef))
            neutral = {**record, 'messages': [dict(message) for message in record['messages']]}
            for message in neutral['messages']:
                if message['role'] == 'user':
                    message['content'] = (f'Implement {function.name}({ast.unparse(function.args)}). '
                                          'Return only Python code in a fenced block.')
            neutral_example = assistant_examples(neutral, backend.tokenizer, manifest['max_length'], pack=True)[0]
            losses = {}
            for label, item in [('original', example), ('signature_only', neutral_example)]:
                with attention_context('math'):
                    loss, count = loss_for(backend.model, item, backend.device)
                losses[label] = loss.item()
            with attention_context('flash'):
                loss, count = loss_for(backend.model, example, backend.device)
            losses['flash'] = loss.item()
            report['records'].append({'episode': key, 'upstream_id': record.get('upstream_id'),
                                      'target_tokens': count, 'loss': losses})
    report['mean_episode_loss'] = {
        label: sum(row['loss'][label] for row in report['records']) / args.count for label in losses}
    report['max_math_flash_loss_difference'] = max(
        abs(row['loss']['original'] - row['loss']['flash']) for row in report['records'])
    report['note'] = ('Random SFT validation episodes, teacher-forced reference loss. Signature-only keeps '
                      'possibly informative function names; this is not proof of semantic understanding or '
                      'execution correctness. Exact prepared input/label identity checked for every original.')
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('records', 'manifest')}))


if __name__ == '__main__':
    main()
