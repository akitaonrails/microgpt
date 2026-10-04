"""Opt-in GPU check: loads our scratch-trained checkpoint; no tools execute.

Run: .venv/bin/python scripts/check_inference.py
This measures execution capacity and cache correctness, not long-context reasoning.
"""
import json
import argparse
import time
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F

from microgpt.attention import cached_attention
from microgpt.inference import NativeBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    for qlen, klen in ((5, 5), (3, 7), (1, 7)):
        torch.manual_seed(4)
        q = torch.randn(1, 4, qlen, 16, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 2, klen, 16, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        mask = torch.arange(klen, device="cuda")[None, :] <= (klen - qlen + torch.arange(qlen, device="cuda"))[:, None]
        with sdpa_kernel(SDPBackend.MATH):
            reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True).transpose(1, 2)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            result, _ = cached_attention(None, q, k, v)
        torch.testing.assert_close(result, reference, atol=0.02, rtol=0.02)
    backend = NativeBackend(args.checkpoint, device="cuda", attention="flash")
    ids = backend.encode([{"role": "user", "content": "Say hello in one word."}], [])
    first = backend.complete(ids, 12, 0, [])
    warm = backend.complete(ids, 12, 0, [])
    assert first.text == warm.text and warm.cached_tokens == len(ids) - 1
    extended = backend.encode([{"role": "user", "content": "Say hello in one word. Then say goodbye."}], [])
    reused = backend.complete(extended, 24, 0, [])
    backend.cache, backend.cached_ids = [{} for _ in backend.model.blocks], []
    cold = backend.complete(extended, 24, 0, [])
    assert reused.text == cold.text and reused.cached_tokens > 0
    print(json.dumps({"cache_equivalence": "passed", "source": backend.source}))
    # Exercise the full context budget without pretending synthetic filler is a benchmark.
    filler = backend.tokenizer.encode("word")[0]
    for length in sorted(set((min(8192, backend.context - 16), backend.context - 16))):
        backend.cache, backend.cached_ids = [{} for _ in backend.model.blocks], []
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tokens = [filler] * (length - len(ids)) + ids
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            result = backend.complete(tokens, 16, 0, [])
        torch.cuda.synchronize()
        kernels = [event.key for event in profile.key_averages() if "flash_attention" in event.key]
        assert kernels, "Expected Flash Attention dispatch"
        print(json.dumps({"prompt_tokens": length, "output_tokens": result.completion_tokens,
                          "seconds": time.monotonic() - start,
                          "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                          "flash_ops": kernels}))


if __name__ == "__main__":
    main()
