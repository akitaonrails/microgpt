"""Inference for our randomly initialized, locally trained GPT checkpoints."""
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .model import GPT, ModelConfig
from .tokenizer import Tokenizer
from .train import precision

from .protocol import prompt_ids, VERSION, tokenizer_identity


def attention_context(mode):
    return sdpa_kernel(SDPBackend.FLASH_ATTENTION) if mode == "flash" else nullcontext()


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    finish_reason: str = "stop"


def sample_token(logits, temperature, generator):
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits.float() / temperature
    threshold = logits.topk(min(40, logits.shape[-1])).values[:, -1:]
    probabilities = logits.masked_fill(logits < threshold, -float("inf")).softmax(-1)
    return torch.multinomial(probabilities, 1, generator=generator)


class NativeBackend:
    """Serve our checkpoint using the same protocol as assistant-only SFT."""
    def __init__(self, checkpoint, device="cpu", attention="sdpa", prefix_cache=True, prefill_chunk=1024):
        self.device = torch.device(device)
        if attention == "flash" and self.device.type != "cuda":
            raise ValueError("Flash Attention requires CUDA")
        self.attention = attention
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = GPT(ModelConfig(**saved["model_config"]))
        self.model.load_state_dict(saved["model"])
        self.model.to(self.device).eval()
        self.tokenizer = Tokenizer.load(Path(checkpoint).parent)
        if self.tokenizer.vocab_size != self.model.config.vocab_size:
            raise ValueError("Missing or incompatible checkpoint tokenizer")
        expected = saved.get("data", {}).get("tokenizer_fingerprint")
        if expected and expected != tokenizer_identity(self.tokenizer):
            raise ValueError("Checkpoint tokenizer fingerprint differs")
        if saved.get("protocol") not in (None, VERSION):
            raise ValueError("Checkpoint uses an unsupported conversation protocol")
        self.context = self.model.config.context
        self.prefix_cache = prefix_cache
        if prefill_chunk < 1:
            raise ValueError("prefill_chunk must be positive")
        self.prefill_chunk = prefill_chunk
        self.cache = [{} for _ in self.model.blocks]
        self.cached_ids = []
        self.source = {"checkpoint": str(checkpoint), "context": self.context, "attention": attention,
                       "parameters": sum(p.numel() for p in self.model.parameters()),
                       "stage": saved.get("settings", {}).get("stage", "pretrain"), "step": saved["step"],
                       "origin": saved.get("origin", {"initialization": "random"})}

    def encode(self, messages, tools):
        return prompt_ids(self.tokenizer, messages, tools)

    @torch.inference_mode()
    def complete(self, ids, max_tokens, temperature, stop, seed=None):
        if len(ids) + max_tokens > self.context:
            raise ValueError(f"Prompt ({len(ids)}) + output ({max_tokens}) exceeds context {self.context}")
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        common = 0
        if self.prefix_cache:
            for old, new in zip(self.cached_ids, ids[:-1]):
                if old != new:
                    break
                common += 1
        if common:
            self.cache = [{name: tensor[:, :, :common] for name, tensor in layer.items()} for layer in self.cache]
        else:
            self.cache = [{} for _ in self.model.blocks]
        self.cached_ids = ids[:common]
        generated = []
        reason = "length"
        try:
            with precision(self.device), attention_context(self.attention):
                for start in range(common, len(ids), self.prefill_chunk):
                    chunk = ids[start:start + self.prefill_chunk]
                    logits, _ = self.model(torch.tensor([chunk], device=self.device), cache=self.cache, last_only=True)
                    self.cached_ids.extend(chunk)
                for index in range(max_tokens):
                    token = sample_token(logits[:, -1], temperature, generator)
                    generated.append(token.item())
                    if token.item() == self.tokenizer.eos or any(s in self.tokenizer.decode(generated) for s in stop):
                        reason = "stop"
                        break
                    if index + 1 < max_tokens:
                        logits, _ = self.model(token, cache=self.cache, last_only=True)
                        self.cached_ids.append(token.item())
        except Exception:
            self.cache = [{} for _ in self.model.blocks]
            self.cached_ids = []
            raise
        finally:
            if not self.prefix_cache:
                self.cache = [{} for _ in self.model.blocks]
                self.cached_ids = []
        text = self.tokenizer.decode(generated)
        for marker in stop:
            text = text.split(marker, 1)[0]
        return Completion(text, len(ids), len(generated), common, reason)
