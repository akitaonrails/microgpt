"""Retain checked curriculum records without regenerating their random names.

Reuses sandbox results only for the exact input hash, and excludes source tests
without assertions. This is filtering existing evidence, not a new execution.
"""
import argparse
import hashlib
import json
from pathlib import Path

from microgpt.curate import normalize_reference_tests, rows


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--checks', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--out-checks', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out_checks.exists():
        parser.error('Use fresh outputs')
    report = json.loads(args.checks.read_text())
    if report['input_sha256'] != sha(args.input) or report['source_sha256'] != sha(args.source):
        raise ValueError('Input identity differs from the executed reference check')
    results = {row['id']: row for row in report['results']}
    if len(results) != report['selected']:
        raise ValueError('Reference check is incomplete or has duplicate IDs')
    keep = set()
    rejected = {}
    for row in rows(args.source):
        identifier = row['seq_id']
        if identifier not in results:
            continue
        if not results[identifier]['passed']:
            rejected[str(identifier)] = 'Renamed reference failed its normalized source tests'
            continue
        try:
            normalize_reference_tests(row['testcase'])
        except (SyntaxError, ValueError) as error:
            rejected[str(identifier)] = str(error)
            continue
        keep.add(identifier)
    if not keep or len(keep) + len(rejected) != len(results):
        raise ValueError('Missing source IDs or no retained references')
    episodes = 0
    with args.out.open('x') as target:
        for line in args.input.open():
            if json.loads(line)['upstream_id'] in keep:
                target.write(line)
                episodes += 1
    filtered = {
        'selected': len(keep), 'passed': len(keep), 'episodes': episodes,
        'input_sha256': sha(args.out), 'source_sha256': report['source_sha256'],
        'image': report['image'], 'parent_report': str(args.checks),
        'parent_report_sha256': sha(args.checks), 'rejected': rejected,
        'results': [row for row in report['results'] if row['id'] in keep],
        'note': 'Filtered original checked records unchanged. One renamed variant per source executed; normalized tests must contain assertions. Not all random names or specification completeness are verified.',
    }
    args.out_checks.write_text(json.dumps(filtered, indent=2) + '\n')
    print(json.dumps({key: filtered[key] for key in ('selected', 'passed', 'episodes')}))


if __name__ == '__main__':
    main()
