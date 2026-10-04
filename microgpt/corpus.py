"""Combine raw code and conversational text for scratch pretraining, preserving groups."""
import argparse
import json
from pathlib import Path
from .protocol import digest, body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with args.out.open('x') as target:
        for path in args.input:
            with path.open() as source:
                for line in source:
                    record = json.loads(line)
                    text = record.get('text')
                    if text is None:
                        text = '\n\n'.join(m['role'] + ':\n' + body(m) for m in record['messages'])
                    target.write(json.dumps({'text': text, 'group': record.get('group') or digest(record),
                                             'source': record.get('source', str(path))}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
