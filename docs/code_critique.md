> **Status (2026-09-01): historical record.** Falsification pass from 2026-08-14,
> before smoke3. Everything under "Fixed" is in the tree (`resolve_model_path`,
> `collect_eos_ids`, resume gating, materialised eval set, fingerprint); the
> "Consciously NOT fixed" list is still accurate for `apod/main.py`. The
> KL-bucket driver (`scripts/bucket_experiment.py`) and `scripts/oracle_kl.py`
> post-date this review and were not covered by it.

# Codebase self-critique (2026-08-14, pre-smoke3)

User-ordered falsification pass over the whole pipeline before smoke3, biased
toward silent-wrongness: "what input or state makes this produce a
wrong-but-plausible result." Method: three independent fresh-context review
agents (determinism/resume, silent-fallbacks/config-semantics,
duplication/dead-code) plus a manual pass over the driver and stages; every
finding below was verified against exact code lines before acting. Ranked by
severity of what the bug would have corrupted.

## Fixed

### Silent experiment corruption (would pass every structural check)

1. **Model-path resolution existed 4x with 3 different existence checks.**
   `train.starting_model_path` tested `config.json` — which retention pruning
   deliberately KEEPS — so a re-run past the prune horizon would load-crash
   deep in `from_pretrained` with an error that looks like corruption;
   `main._model_path` silently fell back to the BASE model for the manifest
   record. Fix: one strict `resolve_model_path` (weights-glob check, error
   names `keep_checkpoints`) in rollout_eval.py, imported by entropy and
   train; the manifest records the checkpoint path unconditionally (honest
   history, resume-safe).

2. **The EOS terminator set was derived 4 different ways, two of which
   disagreed on a reachable input.** train.py's set lacked `<|endoftext|>`
   (declared EOS in the generation config), so a trajectory that finished on
   it would get a SECOND, different EOS appended into the training ids —
   silent training-data corruption, dormant only because cap_hit≈1.0 so far;
   it would have activated exactly when 8192 lets traces finish. Deeper:
   the base 2B repo ships NO generation_config.json and its config.json
   declares a different EOS (`<|endoftext|>`, 248044) than the tokenizer
   (`<|im_end|>`, 248046), while trainer-saved checkpoints declare both — so
   the terminator set silently CHANGED between round 0 and round 1. Fix:
   `collect_eos_ids` unions tokenizer + GenerationConfig + AutoConfig
   (identical at every round); train.py, check_run.py, and
   verify_semantics.py all import it. The appended EOS is now the explicit
   canonical id, not `next(iter(set))`.

3. **`resume=false` desynchronized the stages.** Rollouts regenerated, but
   entropy and train self-skipped on their own done-markers — selection then
   joined STALE entropy scores to FRESH trajectories key-by-key with zero
   errors, and the "trained" checkpoint no longer matched selected.jsonl.
   Fix: both stages now gate their skip on `cfg.resume` and clear their
   outputs when it is false.

4. **Eval problems were re-derived from the live Hub dataset every round.**
   An upstream dataset update (or cache re-download) mid-run would silently
   remap `problem_index -> problem` between rounds — per-round curves would
   compare different problem sets. Fix: the driver materializes
   `pool/eval_problems.jsonl` once; the stage refuses to run without it.

5. **A lost eval shard with a surviving done-marker averaged over half the
   problems and looked normal.** `_merge_eval_summary` now hard-fails unless
   `rows == num_problems x num_samples`.

6. **No structural-config fingerprint.** Changing `num_gpus`,
   `num_rollouts`, `eval.num_samples/num_problems`, `rollout.num_prompts`,
   `seed`, or `max_new_tokens` on a resume silently reinterpreted on-disk
   artifacts (one surviving shard marker "completes" a stage; 12-row groups
   count done at group_size 8; pool round fields go stale). Fix:
   `fingerprint.json` written at first launch; resume under different values
   refuses with the exact diffs.

7. **Parse failure was indistinguishable from wrong answer.** A
   template/thinking regression (a class of bug that already happened once)
   would read as a capability collapse. Eval rows now store
   `has_answer`/`has_boxed`/`gold_parsed`; summaries add `no_answer_rate`
   and `gold_unparsed_problems` (an unparseable GOLD answer zeroes that
   problem for every arm — a dataset defect that was previously invisible).

### Resume robustness (loud failures or wrong metrics on plausible paths)

8. **`write_jsonl` was truncate-and-stream with no atomicity, and
   `selected.jsonl`'s bare existence is a done-marker.** An OOM-kill
   mid-write could leave a line-aligned partial file that resume trusts.
   Fix: atomic tmp + fsync + rename for every whole-file write (also covers
   the metrics.jsonl wholesale rewrite).
9. **A torn final line in any append-mode shard file made every subsequent
   resume crash** until hand-surgery. Fix: resume readers tolerate a torn
   FINAL line only, then atomically rewrite the file clean.
10. **Effective batch was a comment.** `train.effective_batch` is now
    asserted against `world_size x per_device x grad_accum` — changing
    `num_gpus` without retuning accumulation fails loudly instead of
    silently changing the optimizer trajectory.
11. **The driver overwrote any parent `CUDA_VISIBLE_DEVICES`** with logical
    indices (targets someone else's GPUs on a shared box). Fix: logical
    indices map through the parent restriction, with a bounds check.
12. **A typo'd arm burned a full rollout+eval stage before selection
    rejected it.** Arms validated against `apod.selection.ARMS` at startup.
13. Eval-only rounds recorded a fake `rollout_throughput_tok_s` from eval
    tokens (now `None`); train-side tail truncation (drops the ending + EOS)
    is now counted in `summary.json` instead of scrolling by as a warning;
    `needs_entropy` computed once (stage gate and selection merge can no
    longer disagree); the entropy_top4 missing-scores error now names the
    empty-response cause; `cfg.get("resume", True)` unified to `cfg.resume`.

### Dead weight (deleted per the simplicity instruction)

14. `scripts/select_traces.py` deleted (+ its `apod-select` entry point and
    README section): restructure leftover with pre-experiment selection
    policies, a byte-identical duplicate of the entropy math, and the sole
    consumer of `load_student`. Stale exports trimmed
    (`BOXED`/`COLUMNS`/`examples_from_rows`); stale comments fixed (rollout
    header still said 8 rollouts; the presence-penalty docstring argued the
    pre-ADR-0004 position — now points at the ADR).

## Consciously NOT fixed (and why)

- **Rollout chunk seeds depend on pending-list position**, so a mid-round
  crash changes the seed stream for surviving examples. vLLM batching is not
  bitwise-invariant anyway (batch composition changes flip sampled tokens
  through logit numerics), so absolute seeds would not buy true cross-resume
  reproducibility; per-row seeds are honestly recorded. Documented
  limitation, not a statistical bias. The related seed-recording duplication
  (rollout_eval recomputes the generator's derivation to record it) is
  flagged here: any change to the derivation must update both sites.
- **Throughput on partially-resumed rounds is inflated** (all-rows tokens /
  this-invocation wall). Metrics-only, correct on uninterrupted runs; fixing
  it needs stage-side accounting that isn't worth the machinery.
- **GPU probe / duration formatting duplicated in standalone scripts** —
  observability-only; unifying would add a module for telemetry.
- **Hardcoded model ids/paths in standalone measurement scripts**
  (collect_trajectories, verify_presence_penalty, verify_vllm_reload's
  CKPT/BASE). Acceptable for one-shot tools; verify_vllm_reload gets a
  `--checkpoint` argument when it is pointed at real-run checkpoints.
- **Marker fsync under power loss**: atomic writes + torn-tail tolerance
  cover process crashes and OOM-kills; full power-loss durability
  engineering is out of scope for a research pipeline.
- **The HF generation path** (`models/student.py`, `models/teacher.py`, the
  npz `logits` branch) is reachable only from
  `collect_trajectories --with-teacher`. Deletion candidate, but the README
  documents that workflow as the pre-experiment measurement phase — kept
  pending the user confirming that phase is closed.
- **Selection warns (not raises) on unexpected group size**: with the
  fingerprint and `load_complete_rows` the reachable causes are gone; the
  warning stays to permit deliberate partial reruns.
- **`register()` defers to a future vLLM that pre-registers the arch** —
  correct behaviour if upstream ships real support; the version stack is
  pinned, so this cannot change silently.

## Gates already passed: none invalidated

- The EOS round-0/round-1 asymmetry did not touch smoke2's results: ~all
  trajectories cap-truncated (no EOS repair fired), and the 2 natural
  finishes were terminated by the engine's own stop set. verify_semantics'
  loss-identity result used all-truncated rows (no EOS append involved).
- The bitwise tensor verification (248/248 params exact) is independent of
  every finding above.
