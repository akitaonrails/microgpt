"""Random access to JSONL training episodes without loading token lists into RAM."""
from collections import defaultdict
import hashlib
import json


class Episodes:
    def __init__(self, path, expected_hash):
        self.path = path
        self.offsets, self.groups = [], set()
        self.sources = defaultdict(list)
        self.lengths, self.horizons = [], []
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                checksum.update(line)
                item = json.loads(line)
                self.sources[item['source']].append(len(self.offsets))
                self.offsets.append(offset)
                self.groups.add(item['group'])
                self.lengths.append(len(item['input_ids']))
                self.horizons.append(item.get('tool_calls', 0))
        if checksum.hexdigest() != expected_hash:
            raise ValueError(f'Changed data: {path}')

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open('rb') as stream:
            stream.seek(self.offsets[index])
            return json.loads(stream.readline())

    def buckets(self, max_length, min_calls=0, by_length=False):
        buckets = defaultdict(list)
        for source, indices in self.sources.items():
            for index in indices:
                length = self.lengths[index]
                if length > max_length or self.horizons[index] < min_calls:
                    continue
                ceiling = 8192 if length <= 8192 else 16384 if length <= 16384 else 32768
                key = f'{source}@{ceiling}' if by_length else source
                buckets[key].append(index)
        return dict(buckets)
