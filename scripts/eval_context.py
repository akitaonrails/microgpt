"""Measure exact retrieval at 8K/16K/32K, independently of tool-action horizon."""
import argparse
import json
from pathlib import Path
import random
import urllib.request
import urllib.error

from microgpt.chat import ChatRequest, normalize
from microgpt.protocol import prompt_ids
from microgpt.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Only loads the tokenizer beside this checkpoint')
    parser.add_argument('--url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Choose a fresh output file')
    tokenizer = Tokenizer.load(args.checkpoint.parent)
    rng = random.Random(924781)
    results = []
    block = 'Unrelated module note: preserve UTF-8 strings and integer counters.\n'
    for window in (8192, 16384, 32768):
        for position in (0.05, 0.5, 0.95):
            expected = str(rng.randrange(100000, 999999))
            def candidate(count):
                split = int(count * position)
                content = (block * split + f'Build setting REQUIRED_LIMIT = {expected}\n' + block * (count - split)
                           + '\nReturn only the integer assigned to REQUIRED_LIMIT above, with no explanation.')
                messages = [{'role': 'user', 'content': content}]
                normalized, tools = normalize(ChatRequest(model='microgpt', messages=messages))
                return messages, len(prompt_ids(tokenizer, normalized, tools))
            low, high = 0, window // 8
            while low < high:
                middle = (low + high + 1) // 2
                if candidate(middle)[1] <= window - 64:
                    low = middle
                else:
                    high = middle - 1
            messages, tokens = candidate(low)
            request = urllib.request.Request(args.url + '/chat/completions',
                data=json.dumps({'model': 'microgpt', 'messages': messages, 'max_tokens': 64, 'temperature': 0}).encode(),
                headers={'Content-Type': 'application/json'})
            failure = None
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    result = json.load(response)
                text = result['choices'][0]['message'].get('content') or ''
            except (urllib.error.URLError, TimeoutError) as error:
                failure = str(error)
                if isinstance(error, urllib.error.HTTPError):
                    failure += ': ' + error.read().decode()[:2000]
                result, text = {}, ''
            row = {'window': window, 'prompt_tokens': tokens, 'position_fraction': position,
                   'expected': expected, 'answer': text, 'passed': text.strip() == expected,
                   'usage': result.get('usage'), 'error': failure}
            results.append(row)
            args.out.write_text(json.dumps({'checkpoint': str(args.checkpoint), 'kind': 'synthetic exact retrieval, not code comprehension', 'results': results}, indent=2) + '\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
