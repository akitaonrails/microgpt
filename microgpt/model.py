"""Decoder-only Transformer: RMSNorm, rotary positions, causal attention, SwiGLU.

Shapes: B = batch, T = sequence length, C = width, H = attention heads.
PyTorch differentiates the operations; the model itself is spelled out here.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .attention import cached_attention


@dataclass
class ModelConfig:
    vocab_size: int = 257
    context: int = 128
    layers: int = 4
    heads: int = 4
    width: int = 128
    residual_scale: bool = False

    def __post_init__(self):
        if min(self.vocab_size, self.context, self.layers, self.heads, self.width) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width / heads must be an even integer for rotary positions")


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(x.dtype) * self.scale


def rotate(x, offset=0):
    """Rotate pairs of coordinates by a position-dependent angle. x: B,H,T,D."""
    length, dim = x.shape[-2:]
    frequencies = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    angles = torch.outer(torch.arange(offset, offset + length, device=x.device), frequencies)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)

    def forward(self, x, cache=None, offset=0):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
                   for t in (q, k, v)]
        # softmax(Q Kᵀ / sqrt(head_dim)) V; future positions are masked out.
        q, k = rotate(q, offset), rotate(k, offset)
        if cache is None:
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
        else:
            if cache:
                k = torch.cat((cache["key"], k), dim=-2)
                v = torch.cat((cache["value"], v), dim=-2)
            cache.update(key=k, value=v)
            attended, _ = cached_attention(self, q, k, v)
        return self.output(attended.contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.width
        hidden = ((8 * width // 3 + 63) // 64) * 64
        self.attention_norm = RMSNorm(width)
        self.attention = Attention(config)
        self.mlp_norm = RMSNorm(width)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x, cache=None, offset=0):
        x = x + self.attention(self.attention_norm(x), cache, offset)
        normalized = self.mlp_norm(x)
        return x + self.down(F.silu(self.gate(normalized)) * self.up(normalized))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.width)
        self.gradient_checkpointing = False
        self.apply(self._initialize)
        if config.residual_scale:
            # Keep the initial residual variance bounded as depth increases.
            for block in self.blocks:
                for projection in (block.attention.output, block.down):
                    nn.init.normal_(projection.weight, std=0.02 / (2 * config.layers) ** 0.5)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens, targets=None, cache=None, last_only=False, positions=None, loss_chunk=0,
                loss_reduction='token_mean'):
        if loss_reduction not in ('token_mean', 'example_mean'):
            raise ValueError('Unknown loss reduction')
        if cache is not None and (self.training or targets is not None):
            raise ValueError("KV caching is for inference only")
        offset = cache[0]["key"].shape[-2] if cache is not None and cache[0] else 0
        if tokens.size(1) + offset > self.config.context:
            raise ValueError("Input exceeds model context")
        x = self.embedding(tokens)
        for index, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x, cache[index] if cache is not None else None, offset)
        # Reuse the embedding matrix as the output classifier (weight tying).
        if positions is not None:
            if last_only:
                raise ValueError("positions and last_only are mutually exclusive")
            # A batch can supervise different token positions in each example.
            if positions.ndim == 2:
                x = x.gather(1, positions[..., None].expand(-1, -1, x.size(-1)))
            else:
                x = x[:, positions]
        if last_only:
            if targets is not None:
                raise ValueError("last_only cannot be used with training targets")
            x = x[:, -1:]
        x = self.norm(x)
        if loss_chunk:
            if targets is None or targets.shape != x.shape[:2] or loss_chunk < 1:
                raise ValueError("Chunked loss requires matching targets and a positive chunk size")
            # Checkpoint the vocabulary head too: long windows × 200K logits
            # otherwise dominate memory even when attention uses Flash kernels.
            def head_loss(features, labels):
                logits = F.linear(features, self.embedding.weight)
                if loss_reduction == 'example_mean':
                    # Keep short and long SFT episodes equally weighted.
                    losses = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size),
                                             labels.reshape(-1), reduction='none')
                    return losses.view_as(labels).sum(dim=1)
                return F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size),
                                       labels.reshape(-1), reduction="sum")
            total = x.new_zeros((x.size(0),) if loss_reduction == 'example_mean' else (), dtype=torch.float32)
            for start in range(0, x.size(1), loss_chunk):
                features, labels = x[:, start:start + loss_chunk], targets[:, start:start + loss_chunk]
                total = total + (checkpoint(head_loss, features, labels, use_reentrant=False)
                                 if self.training else head_loss(features, labels))
            if loss_reduction == 'example_mean':
                return None, (total / (targets != -100).sum(dim=1)).mean()
            return None, total / (targets != -100).sum()
        logits = F.linear(x, self.embedding.weight)
        loss = None
        if targets is not None:
            if loss_reduction == 'example_mean':
                losses = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size),
                                         targets.reshape(-1), reduction='none').view_as(targets)
                loss = (losses.sum(dim=1) / (targets != -100).sum(dim=1)).mean()
            else:
                loss = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, count, temperature=0.8, top_k=40, eos=None):
        if temperature < 0 or top_k < 0 or count < 0:
            raise ValueError("Sampling arguments must be nonnegative")
        for _ in range(count):
            logits, _ = self(tokens[:, -self.config.context:], last_only=True)
            logits = logits[:, -1].float()
            if temperature == 0:
                next_token = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    threshold = logits.topk(min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, -float("inf"))
                next_token = torch.multinomial(logits.softmax(-1), 1)
            tokens = torch.cat((tokens, next_token), dim=1)
            if eos is not None and (next_token == eos).all():
                break
        return tokens
