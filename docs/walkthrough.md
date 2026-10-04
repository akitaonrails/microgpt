# Follow one prediction

Suppose the training text is `return x + 4`. The byte tokenizer maps each UTF-8
byte to an integer in 0–255. Token 256 means end of document. Portuguese and emoji
work too, but a character may need several bytes. The optional BPE tokenizer learns
frequent byte sequences from the training partition and stores them in
`tokenizer.json`; it usually represents code more compactly.

## From documents to a batch

`prepare.py` deduplicates whole texts and chooses the validation partition using
a stable hash of a group ID (or of the text if no group is supplied). It learns
BPE only from training texts. Both partitions get their own shuffled token stream,
with end-of-document tokens between documents. IDs are stored as little-endian
32-bit integers. This costs four bytes per token on disk.

`TokenStream.batch` chooses random contiguous windows. With a context of four:

```text
stream:   r e t u r n
input:    r e t u
target:   e t u r
```

A batch holds several windows. Windows may cross document boundaries; the model
can see preceding documents in that same partition. End tokens mark the boundary,
but do not reset attention. This simple packing is appropriate for a language
model lesson. Assistant-only supervised training needs additional loss masks.

## Through the model

```mermaid
flowchart LR
    A[Token IDs B × T] --> B[Embeddings B × T × C]
    B --> C[RMSNorm and causal attention]
    C --> D[Residual addition]
    D --> E[RMSNorm and SwiGLU]
    E --> F[Residual addition]
    F --> G[Repeat blocks]
    G --> H[Final RMSNorm]
    H --> I[Vocabulary logits B × T × V]
    I --> J[Cross entropy against next tokens]
```

The embedding table gives each ID a learned vector. Attention projects vectors
into queries, keys, and values. Each head has its own slice of these vectors.
Rotary positions rotate pairs of query/key coordinates by an angle that depends
on position, allowing attention scores to depend on relative position.

A query compares with keys using a scaled dot product. Softmax turns those scores
into weights for a weighted sum of values. A causal mask prevents each position
from reading the answer at later positions. PyTorch's
`scaled_dot_product_attention` computes this operation efficiently. The causality
test changes the future and checks that earlier logits stay unchanged.

RMSNorm controls activation scale. The feed-forward layer applies a learned gate
(`silu(gate(x)) * up(x)`) and projects back to the model width. Residual additions
preserve a direct path through each block. Finally, the embedding matrix is reused
as the vocabulary classifier, saving parameters.

Logits are unnormalized scores. Cross entropy penalizes assigning low probability
to the actual next token. `loss.backward()` applies the chain rule to compute
parameter gradients; AdamW updates the parameters. This is where the scalar
arithmetic and hand-built autograd of Karpathy's original become tensor operations.

## What the training loop adds

Gradient accumulation averages several microbatch losses before an update. It
increases the effective batch without holding all activations at once. The
learning rate warms up, then decays with a cosine curve. Gradient clipping limits
the norm of a single update's gradient. CUDA uses BF16 activations with FP32
parameters and AdamW state; CPU uses FP32 throughout.

Validation uses fixed random windows and no gradients, making successive reported
losses comparable within a run. A checkpoint includes weights, optimizer state,
step number, data metadata, and random states. Saving through a temporary file
protects the last checkpoint if writing the replacement is interrupted.

The log reports natural-log cross entropy per token. `exp(loss)` is token
perplexity. Byte and BPE token losses have different units and cannot be compared
directly. Low text loss also does not guarantee correct generated programs.

## Generating

Load weights and the same tokenizer, encode a prompt, predict a token, append it,
and repeat. Temperature zero picks the largest logit. Positive temperature samples
from a distribution; top-k removes all but the k highest-scoring tokens.
Generation stops at an end token or the requested output length.

The simple `sample` command recomputes the recent context. Once a prompt grows
past the context window, older tokens are dropped. The optional `serve` command
uses per-layer KV caches and prefix reuse; it rejects context overflow instead of
dropping earlier instructions. See `attention.py` and `inference.py` for that path.

## Experiments worth trying

1. Train for 20, 100, and 600 steps in separate output directories. Compare samples
   at temperature zero, then at 0.8. What improves before the loss plateaus?
2. Change the function generator to use different names or operations. Prepare a
   new directory and train new weights. Does the model copy constants correctly?
3. Hold out every multiplication example. Test multiplication after training only
   on addition and subtraction. This is a harder question than template interpolation.
4. Compare bytes and BPE on the same documents. Measure tokens per file, disk size,
   and generated text, rather than comparing token losses directly.
5. Increase width or context separately. Measure peak memory and tokens/second.

The lab keeps one model implementation, separate pretraining and assistant-only
training loops, and single-device execution. The server loads our checkpoints.
Multi-GPU training and reward optimization remain later lessons.
