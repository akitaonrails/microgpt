"""One GPU (or CPU), AdamW, gradient accumulation, validation, portable checkpoints."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import signal
from pathlib import Path
import time
import tomllib

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .data import TokenStream, MixtureStream
from .model import GPT, ModelConfig
from .tokenizer import Tokenizer


def precision(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def evaluate(model, stream, batch_size, batches, device, loss_chunk=128):
    model.eval()
    generator = torch.Generator().manual_seed(1234)
    losses = []
    for _ in range(batches):
        x, y = stream.batch(batch_size, generator, device)
        with precision(device):
            _, loss = model(x, y, loss_chunk=loss_chunk if model.config.vocab_size > 32000 else 0)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path, model, optimizer, step, generator, metadata, settings, origin=None):
    payload = {"model_config": asdict(model.config), "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "step": step,
               "batch_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               "data": metadata, "settings": settings,
               "origin": origin or {"initialization": "random"}}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/demo.toml"))
    parser.add_argument("--data", type=Path, default=Path("data/demo"))
    parser.add_argument("--out", type=Path, default=Path("runs/demo"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, help="Total optimizer steps, including resumed steps")
    parser.add_argument("--stop-at-step", type=int, help="Checkpoint at this absolute step without changing the full learning-rate schedule")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", type=Path)
    source.add_argument("--init-from", type=Path, help="Continue our weights with a new data/window stage and fresh optimizer")
    parser.add_argument("--sequence-length", type=int, help="Training window, bounded by model context; part of resume identity")
    parser.add_argument("--attention", choices=["sdpa", "flash"], default="sdpa")
    parser.add_argument("--max-seconds", type=float, help="Save and stop after this wall-clock training budget")
    args = parser.parse_args()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION) if args.attention == "flash" else nullcontext():
        run_training(args, parser)


def run_training(args, parser):
    config = tomllib.loads(args.config.read_text())
    settings = config["train"]
    if args.steps is not None:
        settings["steps"] = args.steps
    if any(settings[k] < 1 for k in ("steps", "batch_size", "accumulation", "eval_every", "eval_batches")):
        parser.error("Training counts must be positive")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.set_num_threads(4)
    torch.manual_seed(settings.get("seed", 42))
    if device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA training requires BF16 support; use CPU on older GPUs")
        torch.backends.cuda.matmul.allow_tf32 = True
    metadata = json.loads((args.data / "meta.json").read_text())
    model_config = ModelConfig(vocab_size=metadata["vocab_size"], **config["model"])
    sequence_length = args.sequence_length or model_config.context
    if not 1 <= sequence_length <= model_config.context:
        parser.error("sequence-length must fit model context")
    settings["sequence_length"] = sequence_length
    model = GPT(model_config).to(device)
    model.gradient_checkpointing = settings.get("gradient_checkpointing", False)
    if args.attention == "flash" and device.type != "cuda":
        parser.error("Flash Attention requires CUDA")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  betas=(0.9, 0.95), weight_decay=0.1)
    generator = torch.Generator().manual_seed(settings.get("seed", 42))
    start = 0
    origin = {"initialization": "random"}
    if args.init_from:
        from .protocol import tokenizer_identity
        previous = torch.load(args.init_from, map_location="cpu", weights_only=True)
        if previous["model_config"] != asdict(model_config):
            raise ValueError("Stage continuation requires identical model configuration")
        if tokenizer_identity(Tokenizer.load(args.init_from.parent)) != tokenizer_identity(Tokenizer.load(args.data)):
            raise ValueError("Stage continuation requires identical tokenizer IDs")
        model.load_state_dict(previous["model"])
        origin = {**previous.get("origin", {"initialization": "random"}),
                  "parent_checkpoint": str(args.init_from), "parent_step": previous["step"]}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["model_config"] != asdict(model_config) or checkpoint["data"] != metadata:
            raise ValueError("Resume requires identical model configuration and prepared data")
        if checkpoint["settings"].get("sequence_length", model_config.context) != sequence_length:
            raise ValueError("Resume requires identical sequence-length")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["batch_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start = checkpoint["step"]
        origin = checkpoint.get("origin", origin)
    if args.stop_at_step is not None and not start < args.stop_at_step <= settings["steps"]:
        parser.error("stop-at-step must exceed the saved step and fit the full schedule")
    if start >= settings["steps"]:
        parser.error("--steps must exceed the checkpoint step")
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.resume and (args.out / "last.pt").exists():
        parser.error("Output already has a checkpoint; choose another --out or --resume")
    Tokenizer.load(args.data).save(args.out)
    (args.out / "meta.json").write_text(json.dumps(metadata, indent=2))
    def stream(split):
        if metadata.get("sources"):
            return MixtureStream(args.data, split, sequence_length, metadata["sources"])
        return TokenStream(args.data, split, sequence_length)
    train, val = stream("train"), stream("val")
    loss_chunk = settings.get("loss_chunk", 128)
    tokens_per_step = settings["batch_size"] * sequence_length * settings["accumulation"]
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters | {device} | {tokens_per_step:,} tokens/step", flush=True)
    initial = evaluate(model, val, settings["batch_size"], settings["eval_batches"], device, loss_chunk)
    print(f"step {start}: validation loss {initial:.4f}", flush=True)
    started = time.monotonic()
    stopping = []
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopping.append(True))
    for step in range(start, settings["steps"]):
        warmup = settings.get("warmup", 20)
        progress = min(1.0, max(0.0, (step - warmup) / max(1, settings["steps"] - warmup)))
        rate = settings["learning_rate"] * min(1.0, (step + 1) / max(1, warmup))
        rate *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(settings["accumulation"]):
            x, y = train.batch(settings["batch_size"], generator, device)
            with precision(device):
                _, loss = model(x, y, loss_chunk=loss_chunk if model.config.vocab_size > 32000 else 0)
                loss = loss / settings["accumulation"]
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; inspect data and learning rate")
            train_loss += loss.item()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        budget_reached = bool(stopping) or (args.stop_at_step is not None and step + 1 >= args.stop_at_step) or (args.max_seconds is not None and time.monotonic() - started >= args.max_seconds)
        if (step + 1) % settings["eval_every"] == 0 or step + 1 == settings["steps"] or budget_reached:
            val_loss = evaluate(model, val, settings["batch_size"], settings["eval_batches"], device, loss_chunk)
            elapsed = time.monotonic() - started
            record = {"step": step + 1, "train_loss": train_loss, "val_loss": val_loss,
                      "tokens_per_second": (step + 1 - start) * tokens_per_step / elapsed,
                      "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0,
                      "tokens_seen_this_stage": (step + 1) * tokens_per_step, "budget_reached": budget_reached}
            print(json.dumps(record), flush=True)
            with (args.out / "metrics.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
            save_checkpoint(args.out / "last.pt", model, optimizer, step + 1, generator, metadata, settings, origin)
            if budget_reached:
                break


if __name__ == "__main__":
    main()
