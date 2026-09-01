# ADR 0007: Training collation — length-grouped, rank-balanced rows; no padding-free packing

Date: 2026-09-01. Status: proposed (code left on branch worktree-agent-a0b4a8727f85f374c, unmerged: gain ~0.4 min per 100 steps at cap 8192).

## Context

The train stage pads every micro-batch to its longest row (TRL 1.10.0
`DataCollatorForChatML`, left padding). The question was how much compute
that still wastes after length grouping shipped, and whether padding-free
packing (flash-attn varlen over `position_ids`/`cu_seqlens`) would recover
more. Measured offline on real selected rows, replaying transformers 5.15.0's
`get_length_grouped_indices` (sampler batch = per_device x accum per rank,
megabatch = 4 sampler batches) and accelerate 1.14.0's round-robin
`BatchSamplerShard` over 2 ranks (`scripts/verify_row_order.py` reproduces the
mechanics on fake rows):

| rows (cap)                 | micro | random batching pad | length-grouped pad | DDP rank idle (token-weighted) |
|----------------------------|-------|---------------------|--------------------|--------------------------------|
| kl50 kl_mid r1, 544 (8k)   | 2     | 2.1%                | 0.30% (0.15–0.45)  | 0.5%                           |
|                            | 4     | 2.4%                | 0.79%              | 0.7%                           |
| oracle16k kl_mid r1, 512 (16k) | 2 | 14.9%               | 0.45% (0.35–0.56)  | 0.9%                           |
|                            | 4     | 18.8%               | 1.34%              | 3.0%                           |
| oracle16k all r1, 1536 (16k) | 2   | 9.3%                | 0.14%              | 0.3%                           |
|                            | 4     | 11.0%               | 0.41%              | 1.0%                           |

Pad columns are pad / (pad + real) tokens over the whole round; the bracket
is the range over 200 sampler seeds. "Rank idle" is the share of each
optimizer step the lighter rank waits at the all-reduce, with step time taken
proportional to padded tokens. The earlier "9% -> 0.1%" claim holds for the
`all` arm at micro 2; top-4 arms at the 16k cap are bimodal (finished vs
cap-hit rows) and sit at ~0.45% pad plus ~0.9% rank imbalance.

## Decision

Keep pad-to-batch-max collation. `apod.stages.train.order_rows` now produces
the row order explicitly (same algorithm as the trainer's `group_by_length`
sampler, seeded from `train.seed + round`) and interleaves the micro-batches
of each optimizer step so accelerate's round-robin shard gives both ranks the
same token load (0.9% -> 0.24% idle at the 16k cap; ~2 s of a ~320 s step).
The rows summed into each gradient are unchanged; only the rank that computes
a given micro-batch moves.

Rejected:

- **Global length sort** (pad 0.45% -> 0.13%): removes the random megabatch
  membership, making the step sequence deterministic long-to-short. A
  training-dynamics change for ~1 s/step.
- **Padding-free packing.** Not supported by this stack for the hybrid
  Qwen3.5 model, and the frozen teacher would need the identical packed
  layout. Blocking lines in the installed transformers 5.15.0
  `models/qwen3_5/modeling_qwen3_5.py`:
  - `torch_chunk_gated_delta_rule(..., **kwargs)` (line 249) is the torch
    fallback for the 18 Gated DeltaNet layers; `cu_seqlens` arrives in
    `**kwargs` and is never read, so the chunked scan carries recurrent state
    across packed-sequence boundaries. The `fla` kernel that honours
    `cu_seqlens` is not installed (`fla`, `causal_conv1d`, `kernels`,
    `flash_attn` are all absent from the main `.venv`; attention runs on
    sdpa).
  - `causal_conv1d_fn(..., **kwargs)` (line 220) likewise: `F.conv1d` with
    `padding=kernel-1` over the whole packed row leaks 3 tokens across each
    boundary.
  - TRL `experimental/gkd/gkd_trainer.py` `_liger_student_forward` (line
    436) and the teacher forward (line 344) pass only `input_ids` +
    `attention_mask`; a packed layout would need a `compute_loss` override.
- **Dynamic micro-batch by token budget.** Effective batch is asserted at 32
  sequences with fixed `gradient_accumulation_steps`; step time is
  token-bound, not launch-bound, at 8k–16k tokens per row, and micro 4 does
  not fit at the 16k cap.

## Consequences

- Padding is numerically neutral here: labels are -100 on pads, the Liger
  JSD normalises by valid tokens and TRL rescales to the global
  `num_items_in_batch`; Gated DeltaNet projections are bias-free so zeroed
  pad inputs leave the recurrent state untouched, and full attention masks
  pads causally.
- Remaining recoverable collation waste is ~0.7% (pad 0.45% + idle 0.24%):
  not worth further sampler work. The larger training-time lever is that
  the Gated DeltaNet layers run the pure-torch chunk fallback, not a fused
  kernel; that is a dependency decision outside this ADR.
