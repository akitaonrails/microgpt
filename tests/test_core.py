"""A few checks for the mistakes that would invalidate the lesson."""
import tempfile
import unittest
import json
from pathlib import Path

import torch

from microgpt.data import TokenStream
from microgpt.model import GPT, ModelConfig
from microgpt.prepare import demo_documents, prepare
from microgpt.tokenizer import Tokenizer
from microgpt.train import save_checkpoint
from microgpt.attention import cached_attention
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_utf8_round_trip(self):
        tokenizer = Tokenizer()
        text = "def ação(x):\n    return '日本 🐍'"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text) + [tokenizer.eos]), text)

    def test_causality(self):
        torch.manual_seed(1)
        model = GPT(ModelConfig(layers=2, heads=2, width=32)).eval()
        left = torch.randint(0, 257, (2, 16))
        right = left.clone()
        right[:, 8:] = torch.randint(0, 257, (2, 8))
        torch.testing.assert_close(model(left)[0][:, :8], model(right)[0][:, :8])
        cache = [{} for _ in model.blocks]
        with torch.no_grad():
            model(left[:1, :8], cache=cache)
            cached_logits, _ = model(left[:1, 8:], cache=cache)
        torch.testing.assert_close(cached_logits, model(left[:1])[0][:, 8:])

    def test_cached_attention_alignment_and_checkpointed_gradients(self):
        torch.manual_seed(4)
        for length, total in ((5, 5), (3, 7), (1, 7)):
            q = torch.randn(1, 4, length, 16)
            k = torch.randn(1, 2, total, 16)
            v = torch.randn_like(k)
            mask = torch.arange(total)[None, :] <= (total - length + torch.arange(length))[:, None]
            with sdpa_kernel(SDPBackend.MATH):
                reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
                result, _ = cached_attention(None, q, k, v)
            torch.testing.assert_close(result, reference.transpose(1, 2))
        model = GPT(ModelConfig(layers=2, heads=2, width=32))
        tokens = torch.randint(0, 257, (1, 12))
        model(tokens, tokens)[1].backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        model.gradient_checkpointing = True
        model(tokens, tokens)[1].backward()
        for expected, parameter in zip(gradients, model.parameters()):
            torch.testing.assert_close(parameter.grad, expected)

    def test_learning_and_checkpoint(self):
        torch.manual_seed(2)
        model = GPT(ModelConfig(layers=1, heads=2, width=32))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        x = torch.tensor([[1, 2, 3, 1, 2, 3]])
        y = torch.tensor([[2, 3, 1, 2, 3, 1]])
        before = model(x, y)[1].item()
        for _ in range(30):
            optimizer.zero_grad()
            model(x, y)[1].backward()
            optimizer.step()
        self.assertLess(model(x, y)[1].item(), before / 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            generator = torch.Generator().manual_seed(3)
            save_checkpoint(path, model, optimizer, 30, generator, {}, {})
            checkpoint = torch.load(path, weights_only=True)
            loaded = GPT(ModelConfig(**checkpoint["model_config"]))
            loaded.load_state_dict(checkpoint["model"])
            restored_optimizer = torch.optim.AdamW(loaded.parameters(), lr=0.01)
            restored_optimizer.load_state_dict(checkpoint["optimizer"])
            for net, opt in ((model, optimizer), (loaded, restored_optimizer)):
                opt.zero_grad()
                net(x, y)[1].backward()
                opt.step()
            torch.testing.assert_close(model(x)[0], loaded(x)[0], rtol=0, atol=0)
            self.assertEqual(loaded.generate(x, 2, temperature=0).shape, (1, 8))

    def test_split_deduplication_and_shift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared"
            documents = list(demo_documents())
            prepare(documents + documents, path)
            metadata = json.loads((path / "meta.json").read_text())
            self.assertEqual(sum(s["documents"] for s in metadata["splits"].values()), len(documents))
            train = TokenStream(path, "train", 16)
            x, y = train.batch(2, torch.Generator().manual_seed(4), "cpu")
            torch.testing.assert_close(x[:, 1:], y[:, :-1])
            with self.assertRaises(ValueError):
                TokenStream(path, "val", 1_000_000)


if __name__ == "__main__":
    unittest.main()
