"""Frozen development/acceptance families; never import into training generators.

Five wording/name variants per family are correlated: report family scores too.
These authored tasks are not asserted novel relative to public training corpora.
"""
import argparse
import hashlib
import json
from pathlib import Path

# family, signature, specification, reference body, independent assertions.
DEVELOPMENT = [
 ('suffix','text, suffix','Remove suffix only when it is nonempty and occurs at the end; otherwise return text.', 'return text[:-len(suffix)] if suffix and text.endswith(suffix) else text', "assert s.solve('ababa','ba') == 'aba'\nassert s.solve('abc','') == 'abc'\nassert s.solve('abc','x') == 'abc'"),
 ('negatives','values','Return how many numeric items are strictly negative.', 'return sum(v < 0 for v in values)', 'assert s.solve([-2,0,3,-1]) == 2\nassert s.solve([]) == 0'),
 ('rotate','values, amount','Rotate the list left by amount, wrapping around. Return a new list; empty stays empty. Negative amounts rotate right.', 'if not values: return []\namount %= len(values)\nreturn values[amount:] + values[:amount]', 'assert s.solve([1,2,3],4) == [2,3,1]\nassert s.solve([1,2,3],-1) == [3,1,2]\nassert s.solve([],4) == []'),
 ('prefix','words, prefix','Return strings that start with prefix, preserving order.', 'return [w for w in words if w.startswith(prefix)]', "assert s.solve(['ab','ba','ac'],'a') == ['ab','ac']\nassert s.solve([], '') == []"),
 ('product','values','Multiply all numbers; empty input returns 1.', 'result = 1\nfor value in values: result *= value\nreturn result', 'assert s.solve([2,-3,4]) == -24\nassert s.solve([]) == 1\nassert s.solve([0,3]) == 0'),
 ('flatten','rows','Flatten a list of lists by exactly one level, preserving order.', 'return [item for row in rows for item in row]', 'assert s.solve([[1,2],[],[3]]) == [1,2,3]\nassert s.solve([[[1]]]) == [[1]]\nassert s.solve([]) == []'),
 ('frequencies','values','Return a dictionary counting the hashable items.', 'counts = {}\nfor value in values: counts[value] = counts.get(value, 0) + 1\nreturn counts', "assert s.solve(['x','y','x']) == {'x':2,'y':1}\nassert s.solve([]) == {}"),
 ('between','values, low, high','Return numeric items within the inclusive bounds, in input order.', 'return [v for v in values if low <= v <= high]', 'assert s.solve([0,1,3,5],1,3) == [1,3]\nassert s.solve([],0,1) == []'),
 ('alternating','values','Return every other item, starting with index zero.', 'return values[::2]', 'assert s.solve([0,1,2,3,4]) == [0,2,4]\nassert s.solve([]) == []'),
 ('chunks','values, size','Partition a list into consecutive chunks of positive size, including a shorter final chunk.', 'return [values[i:i+size] for i in range(0, len(values), size)]', 'assert s.solve([1,2,3,4,5],2) == [[1,2],[3,4],[5]]\nassert s.solve([],3) == []'),
 ('safe_divide','numerator, denominator','Return the quotient, or None when denominator is zero.', 'return numerator / denominator if denominator != 0 else None', 'assert s.solve(3,2) == 1.5\nassert s.solve(1,0) is None\nassert s.solve(0,4) == 0'),
 ('longest','words','Return the longest string, choosing the first on ties; empty input returns None.', 'return max(words, key=len) if words else None', "assert s.solve(['a','bc','de']) == 'bc'\nassert s.solve([]) is None"),
 ('merge','left, right','Return a new merged dictionary; right values win. Do not modify either input.', 'return {**left, **right}', "a={'x':1}; b={'x':2,'y':3}\nassert s.solve(a,b) == b\nassert a == {'x':1}\nassert b == {'x':2,'y':3}"),
 ('first_match','values, target','Return the first matching index or -1. Equality must support unhashable items.', 'for index, value in enumerate(values):\n    if value == target: return index\nreturn -1', 'assert s.solve([[2],[1],[2]],[2]) == 0\nassert s.solve([],1) == -1'),
 ('difference','values','Return consecutive differences: item i+1 minus item i.', 'return [b-a for a,b in zip(values, values[1:])]', 'assert s.solve([3,1,6]) == [-2,5]\nassert s.solve([3]) == []'),
 ('nonempty','values','Remove only None items; preserve zero, False and empty strings.', 'return [v for v in values if v is not None]', "assert s.solve([None,0,False,'',1]) == [0,False,'',1]\nassert s.solve([]) == []"),
 ('extension','path','Return the final filename extension without its dot, or empty string when there is no extension. Use pathlib semantics.', 'from pathlib import PurePosixPath\nreturn PurePosixPath(path).suffix.lstrip(".")', "assert s.solve('a/b.tar.gz') == 'gz'\nassert s.solve('.env') == ''\nassert s.solve('a') == ''"),
 ('title_words','text','Split on whitespace and return the resulting words as a list, each capitalized.', 'return [word.capitalize() for word in text.split()]', "assert s.solve('  hELLo WORLD ') == ['Hello','World']\nassert s.solve('') == []"),
 ('distance','left, right','Return the sum of absolute coordinate differences between equal-length numeric lists.', 'return sum(abs(a-b) for a,b in zip(left, right))', 'assert s.solve([1,-2],[4,2]) == 7\nassert s.solve([],[]) == 0'),
 ('is_sorted','values','Return whether the list is in nondecreasing order.', 'return all(a <= b for a,b in zip(values, values[1:]))', 'assert s.solve([1,1,2]) is True\nassert s.solve([2,1]) is False\nassert s.solve([]) is True'),
]

ACCEPTANCE = [
 ('deduplicate','values','Remove repeated equal items while preserving first appearance, including unhashable items.', 'result = []\nfor value in values:\n    if value not in result: result.append(value)\nreturn result', 'assert s.solve([[1],[2],[1]]) == [[1],[2]]\nassert s.solve([]) == []'),
 ('running_total','values','Return cumulative sums, one per input item.', 'result = []\ntotal = 0\nfor value in values:\n    total += value\n    result.append(total)\nreturn result', 'assert s.solve([2,-3,5]) == [2,-1,4]\nassert s.solve([]) == []'),
 ('transpose','rows','Transpose a rectangular list of lists; empty input returns an empty list.', 'return [list(column) for column in zip(*rows)]', 'assert s.solve([[1,2],[3,4]]) == [[1,3],[2,4]]\nassert s.solve([]) == []'),
 ('intersection','left, right','Return left items that also occur in right, preserving left order and duplicates. Support unhashable items.', 'return [v for v in left if v in right]', 'assert s.solve([[1],[2],[1]],[[1]]) == [[1],[1]]\nassert s.solve([],[]) == []'),
 ('median','values','Return the median of nonempty numbers without modifying input; average the two middle values for even length.', 'ordered = sorted(values)\nn = len(ordered)\nreturn ordered[n//2] if n % 2 else (ordered[n//2-1] + ordered[n//2]) / 2', 'v=[4,1,3,2]\nassert s.solve(v) == 2.5\nassert v == [4,1,3,2]\nassert s.solve([5]) == 5'),
 ('group_length','words','Group strings by their length in a dictionary, preserving order inside each group.', 'result = {}\nfor word in words: result.setdefault(len(word), []).append(word)\nreturn result', "assert s.solve(['a','bc','d']) == {1:['a','d'],2:['bc']}\nassert s.solve([]) == {}"),
 ('invert','mapping','Invert a dictionary whose values are unique and hashable.', 'return {v:k for k,v in mapping.items()}', "assert s.solve({'a':1,'b':2}) == {1:'a',2:'b'}\nassert s.solve({}) == {}"),
 ('pairwise','values','Return adjacent pairs as tuples.', 'return list(zip(values, values[1:]))', 'assert s.solve([1,2,3]) == [(1,2),(2,3)]\nassert s.solve([]) == []'),
 ('pad','values, length, fill=None','Return a copy padded on the right to at least length. Never truncate longer input.', 'return list(values) + [fill] * max(0, length-len(values))', 'assert s.solve([1],3,0) == [1,0,0]\nassert s.solve([1,2],1) == [1,2]\nassert s.solve([],2) == [None,None]'),
 ('split_once','text, separator','Split on the first occurrence of nonempty separator and return a two-item tuple; absent separator yields (text, empty string).', 'before, _, after = text.partition(separator)\nreturn before, after', "assert s.solve('a:b:c',':') == ('a','b:c')\nassert s.solve('abc',':') == ('abc','')"),
 ('anagram','left, right','Return whether strings are anagrams ignoring whitespace and case using casefold.', 'return sorted("".join(left.casefold().split())) == sorted("".join(right.casefold().split()))', "assert s.solve('Dormitory','Dirty room') is True\nassert s.solve('aab','abb') is False"),
 ('factorial','n','Return factorial of a nonnegative integer, with zero factorial equal to one.', 'result = 1\nfor i in range(2,n+1): result *= i\nreturn result', 'assert s.solve(0) == 1\nassert s.solve(5) == 120'),
 ('gcd','left, right','Return the nonnegative greatest common divisor of two integers.', 'import math\nreturn math.gcd(left, right)', 'assert s.solve(-12,18) == 6\nassert s.solve(0,0) == 0'),
 ('binary','n','Return the base-two representation of a nonnegative integer without a prefix.', 'return format(n, "b")', "assert s.solve(0) == '0'\nassert s.solve(10) == '1010'"),
 ('line_count','text','Return the number of lines using str.splitlines semantics.', 'return len(text.splitlines())', "assert s.solve('a\\nb\\n') == 2\nassert s.solve('') == 0\nassert s.solve('a') == 1"),
 ('json_keys','text','Parse a JSON object and return its keys as a sorted list.', 'import json\nreturn sorted(json.loads(text))', "assert s.solve('{\"b\":1,\"a\":2}') == ['a','b']\nassert s.solve('{}') == []"),
 ('basename','path','Return the last component using POSIX path semantics.', 'from pathlib import PurePosixPath\nreturn PurePosixPath(path).name', "assert s.solve('/a/b.txt') == 'b.txt'\nassert s.solve('a/b/') == 'b'"),
 ('leap','year','Return whether year is a Gregorian leap year.', 'return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)', 'assert s.solve(2000) is True\nassert s.solve(1900) is False\nassert s.solve(2024) is True'),
 ('seconds','value','Convert a nonnegative total number of seconds to an (hours, minutes, seconds) tuple. Hours may exceed 23.', 'hours, rest = divmod(value, 3600)\nminutes, seconds = divmod(rest, 60)\nreturn hours, minutes, seconds', 'assert s.solve(3661) == (1,1,1)\nassert s.solve(90000) == (25,0,0)'),
 ('strip_comments','lines','For each string remove everything from the first # onward, then strip surrounding whitespace. Preserve empty results.', 'return [line.partition("#")[0].strip() for line in lines]', "assert s.solve([' a # x','# hi','b']) == ['a','','b']\nassert s.solve([]) == []"),
]


def cases(split):
    families = DEVELOPMENT if split == 'development' else ACCEPTANCE
    variants = ['Implement {signature}. {spec}', 'Write Python function {signature}: {spec}',
                'I need {signature}. Requirements: {spec}', 'Create {signature} in Python. {spec}',
                'Please implement {signature} with this behavior: {spec}']
    for family, arguments, spec, body, checks in families:
        for index, wording in enumerate(variants):
            name = f'{family}_{index}'
            signature = f'{name}({arguments})'
            yield {'id': name, 'family': family, 'split': split,
                   'prompt': wording.format(signature=signature, spec=spec),
                   'reference': f'def {signature}:\n' + '\n'.join('    '+line for line in body.splitlines())+'\n',
                   'tests': checks.replace('s.solve(', f's.{name}(')+'\n'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    manifest={'note':'100 variants per split, 20 families per split. Family sets are disjoint. Not asserted uncontaminated relative to public training.'}
    for split in ('development','acceptance'):
        path=args.out/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in cases(split)))
        manifest[split]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'tasks':100,'families':20}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__': main()
