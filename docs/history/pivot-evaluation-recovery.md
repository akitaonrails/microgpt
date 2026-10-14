# 2B evaluation interruption and recovery

The 337M scratch foundation completed 2,000,027,648 tokens at step 61,036.
Final validation loss was 1.32607. Raw completion scored 0/100 development tasks.
The subsequent 500-update code SFT checkpoint was saved successfully, but its
constrained evaluation server failed during initialization: ToolGrammar supported
bytes and o200k but rejected our new ByteLevel BPE tokenizer.

This is an integration bug, not an instruction-following score. The earlier
preflight exercised training and raw inference, missing the constrained serving
boundary. The fix supplies the BPE vocabulary to XGrammar as BYTE_LEVEL rather
than RAW; decoding individual tokens as text would damage partial UTF-8 bytes.
A regression test accepts Unicode paths token by token and rejects unknown tools.

Recovery reuses the saved step-500 checkpoint and optimizer. Results go to
runs/pivot-v1/coding-2b-recovery, preserving the original failed run. It evaluates
step 500, continues to 1000, and evaluates again under the original recipe.
The original foundation controller remains failed; the recovery is a separate
process. Its results must be reconciled against the original spending gates
before continuing foundation training. No new capability or promotion is claimed.

## Recovered result

Both evaluations completed: **24/100 at 500 SFT updates; 39/100 at 1,000**.
The latter exceeds the previous best 24/100 by 15 tasks. It is a development-set
improvement, not fresh-test validation or evidence of agent readiness. Syntax was
valid in 98/100 outputs and requested parameter names matched in 86/100.

The original 2B spending gate (at least 10/100) passed. The controller now supports
`--resume-after-2b`, checking plan/config/data hashes, complete recovered scores,
and the preserved foundation checkpoint hash before continuing. The failed status
is archived. Continuation restores the foundation optimizer/RNG toward 5B, with
75% GPU pacing; it does not continue from instruction-trained weights. At 5B,
the original gate now requires at least 42/100 (three above the 2B score) to extend
to 10B. No gate was relaxed in response to results.
