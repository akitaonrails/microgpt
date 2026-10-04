"""Load a checkpoint and continue a prompt. Temperature 0 selects the most likely token."""
import argparse
from pathlib import Path
import torch
from .model import GPT, ModelConfig
from .tokenizer import Tokenizer
from .train import precision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/demo/last.pt"))
    parser.add_argument("--prompt", default="def add_42(x):\n    return")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = GPT(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    tokenizer = Tokenizer.load(args.checkpoint.parent)
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("Missing or incompatible tokenizer beside checkpoint")
    tokens = torch.tensor([tokenizer.encode(args.prompt) or [tokenizer.eos]], device=device)
    with precision(device):
        result = model.generate(tokens, args.tokens, args.temperature, args.top_k, tokenizer.eos)
    print(tokenizer.decode(result[0].tolist()))


if __name__ == "__main__":
    main()
