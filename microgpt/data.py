"""Memory-mapped token streams; targets are inputs shifted one position right."""
from pathlib import Path
import numpy as np
import torch


class TokenStream:
    def __init__(self, directory, split, context):
        self.tokens = np.memmap(Path(directory) / f"{split}.bin", dtype="<u4", mode="r")
        self.context = context
        if len(self.tokens) <= context:
            raise ValueError(f"{split} has {len(self.tokens)} tokens; needs more than context={context}")

    def batch(self, size, generator, device):
        starts = torch.randint(len(self.tokens) - self.context, (size,), generator=generator)
        windows = np.stack([self.tokens[int(i):int(i) + self.context + 1] for i in starts])
        tokens = torch.from_numpy(windows.astype(np.int64)).to(device)
        return tokens[:, :-1], tokens[:, 1:]


class MixtureStream:
    """Choose one source per microbatch; equal windows make weights token weights."""
    def __init__(self, directory, split, context, sources):
        self.streams = [TokenStream(Path(directory) / item["path"], split, context) for item in sources]
        self.weights = torch.tensor([item["weight"] for item in sources], dtype=torch.float64)
        if not torch.isfinite(self.weights).all() or (self.weights <= 0).any():
            raise ValueError("Mixture weights must be finite and positive")

    def batch(self, size, generator, device):
        index = int(torch.multinomial(self.weights, 1, generator=generator))
        return self.streams[index].batch(size, generator, device)
