# Local GPU and Runpod

> Historical recipes and measurements. See the [October 14 closeout report](../report/experiment-closeout.md) for the final model, data inventory, timing and capability results.

Start on the RTX 5090. This machine reported 32,607 MiB total VRAM, with roughly
28,502 MiB free at inspection. That is ample for the teaching model and leaves
room for substantially larger experiments. Renting an H100 is optional.

## Presets

| Preset | Parameters | Context | Effective tokens/update | Purpose |
| --- | ---: | ---: | ---: | --- |
| `demo.toml` | 0.886M | 128 | 2,048 | Offline lesson; bytes |
| `rtx5090.toml` | 29.89M | 512 | 32,768 | Modest single-GPU experiments; 8,192-token BPE |
| `h100.toml` | 91.25M | 1,024 | 65,536 | Larger single-GPU experiment; 8,192-token BPE |

Parameter counts depend on vocabulary size. The H100 preset is not exclusive to
H100 hardware: it may fit locally too. Measure before renting. These presets are
starting configurations, not quality guarantees or compute-optimal recipes.

Training holds FP32 weights, gradients, and two AdamW moments: approximately
16 bytes per parameter before activations, temporary buffers, and allocator
reserve. Logit memory also grows with batch × context × vocabulary. Reducing
`batch_size` is the first memory adjustment; increase `accumulation` if you want
the same effective token batch. Longer contexts and larger vocabularies add costs
that parameter count alone misses.

CUDA training uses BF16 and a PyTorch attention kernel. The tested installation
is PyTorch 2.7.1+cu128, CUDA runtime 12.8, driver 610.57.04. The CUDA toolkit need
not be separately compiled for this project. Check the current
[official wheel/driver guidance](https://pytorch.org/get-started/previous-versions/)
when choosing a different environment. No custom FlashAttention build is required.

## Measure before a long run

```sh
.venv/bin/python -m microgpt.train \
  --config configs/rtx5090.toml --data data/code --out runs/pilot --steps 100
```

Read `tokens_per_second`, `peak_vram_gb`, and validation loss from the log.
Peak memory is PyTorch's allocated memory, not total device usage. Throughput
includes validation and earlier checkpoint writes, but excludes the initial
validation and the current checkpoint write. Very short pilots underrepresent
steady-state performance.

The default RTX run processes 163.84M tokens; the H100 run processes 655.36M.
Those are exposures, not distinct corpus tokens. Repeating a 5,000-record dataset
thousands of times can overfit. Compare the prepared training token count with
`steps × batch_size × context × accumulation` before scheduling a run.

Estimate training hours as `target_tokens / measured_tokens_per_second / 3600`.
Multiply by the live rental quote, then add startup, downloads, storage, and time
spent idle. A pilot at one model/context/vocabulary does not predict another.
No H100 price or speedup is assumed here.

## Desktop headroom

GPU training now defaults to a **75% soft duty cycle** so a local display GPU has
breathing room for the browser/compositor. Both pretraining and SFT wait for their
queued CUDA work to complete and yield between microbatches, optimizer updates and
validation examples. This changes wall time, not the batch, optimizer or token
schedule. It does not reserve a fixed share of GPU cores or VRAM, and a single long
kernel can still cause a latency spike.

Use `MICROGPT_GPU_DUTY_CYCLE=0.6` for more desktop headroom. On a dedicated headless
GPU, explicitly use `MICROGPT_GPU_DUTY_CYCLE=1` for unrestricted training. The
environment is inherited by multi-stage runners; `nice -n 10` can additionally
lower their CPU priority. The selected duty cycle is printed and logged with
training metrics. Changing it when resuming preserves optimizer/RNG state.

## Run on one Runpod H100 80 GB

1. Create a GPU Pod with one H100 and a CUDA-capable PyTorch image. Attach a network
   volume at creation if you want it to survive Pod termination. Check the live
   hourly rate and storage charges in Runpod's console.
2. Upload this repository and prepared `data/code` into `/workspace/microgpt`.
   Exclude `.venv`, caches, and unneeded run checkpoints. Use Runpod's displayed
   SSH host/port with `scp` or `rsync`; avoid hard-coding endpoint credentials.
3. In the Pod, install the tested environment and run:

```sh
cd /workspace/microgpt
uv venv --python 3.12
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e '.[data]'
MICROGPT_GPU_DUTY_CYCLE=1 .venv/bin/python -m microgpt.train \
  --config configs/h100.toml --data data/code --out runs/h100 --stop-at-step 100
```

This stops after 100 updates while preserving the config's full 10,000-update
schedule. To continue, keep that same schedule and checkpoint:

```sh
MICROGPT_GPU_DUTY_CYCLE=1 .venv/bin/python -m microgpt.train \
  --config configs/h100.toml --data data/code --out runs/h100 \
  --resume runs/h100/last.pt --steps 10000
```

For a fresh full run, use a new output directory and omit `--stop-at-step 100`.
A run created with `--steps 100` has a different schedule and cannot be resumed
as the same experiment with 10,000 steps. Use `--init-from` and a fresh output
directory if intentionally starting a new optimizer/schedule stage from its weights.

Run in a persistent terminal such as `tmux` if disconnecting SSH. Download the
whole run directory, configs, prepared metadata, and source manifest before
terminating. Locally, `sample --checkpoint ... --device cpu` can inspect the
result even without the same GPU.

To move an *existing* local run to the H100, use the local run's original config;
changing to `h100.toml` changes the architecture and cannot resume those weights.
The trainer deliberately rejects that mismatch. This is single-device code;
`torchrun` would start independent trainers, not distributed training.

Runpod distinguishes container storage, Pod volume storage, and network volumes.
A path named `/workspace` alone does not establish that it survives termination.
See [Runpod storage lifetimes](https://www.runpod.io/blog/where-did-my-files-go-a-straight-guide-to-runpod-storage)
and [Pod management](https://docs.runpod.io/pods/manage-pods). Verify your files and
terminate the GPU resource when finished; persistent storage may remain billable.

No Pod was rented or cloud execution verified during this implementation.

## Long-context extension

The new `agent-8k.toml` and `agent-32k.toml` presets retain 91.25M parameters with
an 8K vocabulary and use activation checkpointing. Both were smoke-tested locally
with `--attention flash`. The 32K update peaked at 5.606 GB of PyTorch allocation;
that excludes other processes and reserved memory. Dataset partitions must each
be longer than the configured context. See [OpenCode setup](opencode.md) for the
scratch-checkpoint serving and the verification record for measured inference.

The current student preset uses eight layers at width 384 and a 32K ceiling.
With o200k_base it has exactly 90,969,600 parameters. Start with 2K training
windows via `--sequence-length 2048`; tokenizer vocabulary drives much of the
parameter/logit cost. Keep the same config when moving this run to an H100.
Install `.[data,tokenizer,tune,serve]` for the complete scratch assistant pipeline.
