"""Compare direct and thinking answers at equal output budgets on fresh problems.

Scores final integers, never verbosity. Family 3 is absent from our procedural
training set. This is a narrow program-tracing evaluation, not a reasoning claim.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from microgpt.reasoning_data import trace_problem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--tasks', type=int, default=12)
    parser.add_argument('--budgets', type=int, nargs='+', default=[256, 1024])
    args = parser.parse_args()
    if args.out.exists() or args.tasks < 4 or min(args.budgets) < 1:
        parser.error('Use a fresh output file, at least four tasks and positive budgets')
    results = []
    report = {'checkpoint': args.label, 'note': 'Held-out inputs; family 3 is a held-out procedural template. No model code is executed.', 'results': results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.tasks):
        problem = trace_problem(900000 + index, family=index % 4)
        for budget in args.budgets:
            for effort in ('none', 'high'):
                row = {'task': index, 'family': problem['family'], 'expected': problem['answer'],
                       'effort': effort, 'budget': budget, 'correct': False}
                payload = {'model': 'microgpt', 'messages': [{'role': 'user', 'content': problem['question']}],
                           'reasoning_effort': effort, 'max_tokens': budget, 'temperature': 0, 'seed': 42}
                start = time.monotonic()
                try:
                    request = urllib.request.Request(args.url.rstrip('/') + '/chat/completions',
                                                     data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(request, timeout=180) as response:
                        response = json.load(response)
                    choice = response['choices'][0]
                    message = choice['message']
                    row.update(answer=message.get('content'), reasoning=message.get('reasoning_content'),
                               finish_reason=choice['finish_reason'], usage=response['usage'])
                    row['correct'] = (message.get('content') or '').strip() == problem['answer']
                except Exception as error:
                    row['error'] = str(error)
                row['seconds'] = time.monotonic() - start
                results.append(row)
                args.out.write_text(json.dumps(report, indent=2) + '\n')
    report['scores'] = []
    for budget in args.budgets:
        for effort in ('none', 'high'):
            for held_out in (False, True):
                selected = [r for r in results if r['budget'] == budget and r['effort'] == effort and (r['family'] == 3) == held_out]
                report['scores'].append({'budget': budget, 'effort': effort, 'held_out_template': held_out,
                                         'correct': sum(r['correct'] for r in selected), 'total': len(selected)})
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['scores'], indent=2))


if __name__ == '__main__':
    main()
