# Active OPD — code review

Scope: whole repository (`aopd/`, `scripts/`, `configs/`, `tests/`), read against
`docs/idea.md` and `todo.md`.

Method: nine independent reviewers over separate dimensions (loss math, token
alignment, trainer, rollout collection, verifier/selector, data/eval, scripts and
Hydra wiring, research design, tests), each followed by an adversarial verifier
required to reproduce or refute every claim by executing code. 101 findings
raised, 100 survived verification, 1 refuted. Claims below marked *(reproduced)*
were re-checked directly against the repo, its `.venv`, and the committed
artifacts in `outputs/`.

Key evidence reproduced during the review:

- `E_{y~pi_theta}[grad k3] == grad KL(teacher||student)` to 1e-17 — the default
  loss descends the *forward* KL. `k1`'s expected gradient is exactly zero.
- AdamW at `lr=1e-5` on bf16 parameters leaves 84.9% of entries bitwise
  unchanged after 16 steps (0.0% in fp32); on the real Qwen3-1.7B, 90.1%.
- `outputs/actual-model-benchmark/`: standard took 8 optimizer steps, active took
  2, on byte-identical rollouts (`responses_sha256` equal).
- Both rollouts in that run's Active pool were mathematically correct
  (`(3, pi/2)`), labelled `wrong` only because the normalizer cannot equate
  unicode/LaTeX forms. 9 of 10 realistic MATH-500 answer pairs are misjudged.
- 7 of 8 pilot traces contain no `</think>` and no `\boxed` — all 8 hit the
  1024-token cap.
- `open-thoughts/OpenThoughts-114k` default split, first 40 records: 40/40 are
  competitive-programming prompts; columns are `['system', 'conversations']`
  with no answer field.

---

# AOPD code review — what is actually broken

## 1. Verdict

The scaffolding is sound (Hydra composition, model wrappers, JSONL logging, a real 4090 pilot that ran end to end), but every one of the four things the experiment's conclusion depends on is independently broken, and each one alone is sufficient to make every number in `outputs/` meaningless: the optimizer cannot move most of the weights, the loss descends the wrong divergence, the training corpus is competitive programming rather than mathematics, and the verifier labels correct answers wrong. The single most important defect is `aopd/train/trainer.py:189-209`: the optimizer is bound directly to bf16 parameters with no fp32 master copy, and at `lr=1e-5` **90.1% of Qwen3-1.7B's parameters are bitwise identical after 16 real optimizer steps** — measured on the actual model with the repo's own `_build_optimizer`. Everything downstream of that is measuring an untrained student. Second-most important: the default `k3` path at `aopd/losses/opd.py:180` backpropagates `∇KL(teacher‖student)`, not the `∇KL(student‖teacher)` the whole project is defined around, so the repo does not currently implement on-policy reverse-KL distillation at all. The research contribution — the acquisition score — is not merely unimplemented; as shipped, the "Active OPD" arm *is* baseline #3 from `docs/idea.md:146-147` with the sign flipped, so the headline arm and one of its own controls are the same estimator.

---

## 2. Blocking correctness bugs

### B1 — Training is a near no-op: bf16 weights with no fp32 master copy
`aopd/train/trainer.py:189-209`, `configs/model/student/qwen3_1_7b.yaml:3`

`_build_optimizer` constructs `AdamW8bit`/`AdamW` straight over `self.student.model.parameters()`, which are bf16 (`ModelLoadOptions.dtype` → `from_pretrained` at `aopd/models/common.py:70`). There is no fp32 shadow copy anywhere; `_configure_amp` (`trainer.py:408-479`) only sets `torch.autocast`, which changes op compute dtype and never creates master weights. bf16 has an 8-bit mantissa: 1 ulp at |w|≈0.022 is 8.6e-5, half-ulp 4.3e-5, while an Adam step at lr=1e-5 is ~1e-5. `p.add_(update)` rounds to a no-op for most entries.

**Failure**: on the real `Qwen/Qwen3-1.7B`, 16 steps with the repo's own optimizer over 4.06M sampled elements across all 311 tensors → `frac_unchanged = 0.9007`, `mean|Δw| = 5.98e-6` vs `0.0000 / 5.06e-5` for the identical fp32 run. Per-tensor: `embed_tokens` 0.920 unchanged, layer-0 `q_proj` 0.906, every LayerNorm ≈ 1.000. The ~10% that do move move too far, so the update is grossly distorted, not merely absent. Loss decreases, `optimizer_steps` counts up, checkpoints are written, and `post_training` accuracy ≈ `pre_training` accuracy for both arms.

**Fix**. Two reviewers proposed `dtype: float32` — that is wrong on this card and I am overriding it: fp32 weights (6.8 GiB) + fp32 grads (6.8) + 8-bit Adam state (3.4) + NF4 teacher (2.5) ≈ 19.5 GiB before the ~2.3 GiB of logits I measured at B=1, and the committed run already peaked at 20.7 GiB. Use a stochastic-rounding / Kahan-summation optimizer (`optimi.AdamW(..., kahan_sum=True)`) which keeps bf16 storage, or drop to a 0.6B student, or use LoRA. Regardless, add a startup assertion:

```python
before = {n: p.detach().clone() for n, p in list(model.named_parameters())[:4]}
# ... N steps ...
moved = mean(float((p != before[n]).float().mean()) for n, p in ...)
if moved < 0.5: raise RuntimeError(f"only {moved:.1%} of sampled params changed")
```

### B2 — The default loss descends the forward KL
`aopd/losses/opd.py:180` (estimator), `:275-286` (backprop path), `configs/estimator/sampled_token.yaml:2`

`reverse_kl_estimator` returns the k3 **value** `exp(-r) - 1 + r` with `r = s - t`, `t` detached, and the trainer backprops through it pathwise. `d k3/ds = 1 - e^{-r} = 1 - q(y)/p(y)`, so

```
E_{y~p}[(1 - q/p) ∇log p] = Σ_v (p_v - q_v) ∇log p_v = ∇_θ ( -Σ_v q_v log p_v ) = ∇_θ KL(q‖p)
```

k3 is an unbiased estimator of the KL *value*; its pathwise derivative is not the derivative of that value. The missing score-function term `∇E_p[k3] - E_p[∇k3]` is exactly the gap. Verified by exact float64 enumeration through the real API: `‖E_p[∇k3] - ∇KL(teacher‖student)‖ = 6.9e-17` versus `1.67e-01` against the intended reverse KL. On real logits (Qwen3-1.7B rollouts scored by the NF4 4B, 1024-2304 response tokens): `cos(E[∇k3], ∇KL_rev)` mean 0.51-0.60, **2.5-3.4% of response tokens have negative cosine** — the update provably increases the reverse KL there — and the weighting is inverted at the tail (the `r=+10.55` token gets k3 weight 1.0000; the top-1% highest-signal tokens carry 26% of the correct gradient mass but 11% under k3). Under model misspecification the optima genuinely differ: on a constrained 6-token family, reverse-KL optimum `[0.002, 0.002, 0.249×4]` vs k3-loss optimum `[0.492, 0.492, 0.004×4]`, L1 = 1.96.

Two related traps on the same dispatch. `estimator: k1` (`opd.py:175`) has **exactly zero expected gradient** (`‖E_p[∇]‖ = 5.1e-17`) because the teacher is detached — the natural "does the estimator matter?" ablation would train on pure noise. And the only correct implementation in the repo, `compute_policy_gradient_opd_loss` (`opd.py:300`, verified `‖E_p[∇] - ∇KL_rev‖ = 1.2e-16`), has zero call sites; its docstring at `opd.py:309-311` claims it shares estimator semantics with the default path when the two optimize different divergences.

**Fix**, in order of preference:
1. Route the trainer to `compute_policy_gradient_opd_loss` and delete the k3-as-loss branch (demote k3 to a no-grad logged diagnostic). Thread `reduction` through `masked_mean(values, response_mask, reduction=...)` at `opd.py:324`.
2. Or switch the sampled path to k2, `0.5 * log_ratio.square()` (`opd.py:178`) — `E_p[r ∇log p] = ∇KL(p‖q)` exactly, verified to 8.3e-17. Stop describing it as "the squared approximation"; here it is the correct REINFORCE estimator.
3. Exact full-vocab `Σ_v p_v(log p_v - log q_v)` removes the estimator bias *and* B10 at once — but note a contradiction between reviewers: naive exact full-vocab costs **more** memory than the shipped path (2.90 vs 1.74 GiB at T=1024; 5.80 vs 3.48 at T=2048). The claimed 2× saving comes entirely from chunking + activation checkpointing, which is orthogonal — a chunked+checkpointed *sampled* k3 measures identically (0.869/1.740 GiB). Choose exactness on its merits, not memory.

Ship this regression test with the fix; it fails today and would have caught B2, k1, and B9 in one assertion:

```python
def test_expected_grad_equals_analytic_reverse_kl(estimator):
    V = 4; theta = torch.randn(V, requires_grad=True); teacher = torch.randn(V)
    log_q, log_p = torch.log_softmax(teacher, -1), torch.log_softmax(theta, -1)
    (want,) = torch.autograd.grad((log_p.exp() * (log_p - log_q)).sum(), theta)
    probs = log_p.exp().detach(); got = 0
    for v in range(V):                       # exact expectation over y ~ student
        t = theta.detach().clone().requires_grad_(True)
        l = compute_opd_loss(t.view(1,1,V), teacher.view(1,1,V), torch.tensor([[v]]),
                             torch.ones(1,1,dtype=torch.bool),
                             OPDLossConfig(estimator=estimator, clamp_log_ratio=None))
        got = got + probs[v] * torch.autograd.grad(l, t)[0]
    assert torch.allclose(got, want, atol=1e-5)
```

### B3 — The verifier turns correct answers into the entire Active-OPD training pool
`aopd/data/answers.py:73-92` (`normalize_answer`), `:174-176` (byte equality), `:42` (`is_retained_for_active_opd == (outcome == "wrong")`), `aopd/train/selector.py:56-76`

`normalize_answer` strips only `−`, `\left/\right`, `\,`, `$`, `\text{}`, `\mathrm{}`, whitespace, trailing punctuation, and thousands separators. No `\dfrac→\frac`, no `\frac{a}{b}↔a/b`, no unicode `π→\pi`, no decimal/fraction canonicalisation. Because `VerifiedWrongSelector` retains exactly `outcome == "wrong"`, a false negative is not merely dropped — it is *promoted into the training set* as a "student mistake", and it simultaneously deflates the headline accuracy.

**Failure, from your own committed artifact**: `outputs/actual-model-benchmark/raw/active.jsonl` has outcomes `{malformed: 5, wrong: 2, correct: 1}`. Both `wrong` records are `normalized_predicted = "(3,π/2)"` vs `normalized_reference = "(3,\frac{\pi}{2})"` — the same answer. `results.json` shows `retained_rollouts: 2`. **100% of the Active-OPD pool for that run was mathematically correct answers mislabeled by the normalizer**, and the teacher was asked to "correct" them. Independently: 9/17 and 11/13 realistic MATH-500 pairs get the wrong label (`\dfrac{1}{2}` vs `\frac{1}{2}`, `0.5` vs `\frac{1}{2}`, `2\sqrt3` vs `2\sqrt{3}`, `1000` vs `1{,}000`, `x=5` vs `5`, `5^\circ` vs `5`, `\mbox{}` vs `\text{}`). ~30% of MATH-500 references contain LaTeX/unicode/slash/decimal, i.e. are format-sensitive.

Two **false positives** in the same function, both confirmed: `normalize_answer(".5") == "5"` (the `strip(".,;:")` at `answers.py:89` eats the leading dot), and `normalize_answer("100,200,300") == "100200300"` (the `fullmatch` at `:90` accepts multi-group lists, so a 3-element list equals a 9-digit integer).

**Fix**: replace `verify_exact_answer`'s comparator with `math_verify` (`parse` + `verify`) or the Hendrycks `is_equiv`. Add a third outcome `unverified` for anything the comparator cannot decide, retained by *no* selector — never default to `wrong`. Tighten `:90` to a single-group check and drop the leading-`.` strip. Add the parametrized regression table over the pairs above. Then re-label the committed traces offline (zero GPU) and report the old gate's false-negative rate — that number decides whether any selection result in this repo means anything.

### B4 — The training corpus is competitive programming labelled with prose fragments
`scripts/run_opd_benchmark.py:265-273`, `aopd/data/datasets.py:70-81`, `aopd/data/answers.py:137-142`, `scripts/run_opd_benchmark.py:230`

`_load_datasets` streams `open-thoughts/OpenThoughts-114k` from record 0 with no shuffle, no domain filter and no `config_name` (`OpenThoughtsConfig` at `aopd/data/openthoughts.py:13-19` has no such field; `configs/data/openthoughts.yaml`'s `prompt_fields`/`answer_fields`/`limit` are never passed). That split is **sorted by domain** and has only `['system','conversations']` columns — no answer field — so every reference falls through to `extract_final_answer(solution)`, i.e. the `(?:final\s+answer|answer)\s*(?:is|=|:)` regex over a DeepSeek-R1 chain of thought. `_load_valid_examples` accepts any non-empty string.

**Failure, reproduced through the exact production path**: `--train-samples 32` scans 65 records, skips 33, and returns 32 examples of which **32/32** are `"Generate an executable Python function generated from the given prompt..."`. Over the first 3000 records: 3000/3000 codegen, 1613 non-empty scraped references, **zero** from `\boxed`. Sample "reference answers": `"'yes' if"`, `"expected to handle all cases, including large ones"`, `'"13-12-2013", which occurs three times. So how does this happen?'`. Your own `outputs/qwen3-multisample-benchmark/config.json` records exactly this (`records_scanned=65, records_skipped=33, loaded_examples=32`). Consequence: student `\boxed{}` math answers can never match, `correct ≈ 0`, so `standard`'s `("correct","wrong")` pool collapses onto `active`'s `("wrong",)` and the two arms become the same run — while `results.json` looks well-formed. Math records only start around row ~21000.

**Fix**: switch to a corpus with a ground-truth answer column (`open-r1/OpenR1-Math-220k`, MATH train levels 3-5). Note the OpenThoughts `metadata` config will *not* work as a drop-in: its columns are `problem/deepseek_reasoning/deepseek_solution/ground_truth_solution/domain/source/...`, none of which `example_from_record`'s field tuples look for, so every record would yield `reference_answer=None`. Make `example_from_record` **raise** when no answer column exists rather than scraping, or accept the fallback only when `extracted.source == "boxed"` *and* the answer is a short atom. Add a startup assertion that ≥95% of drawn references parse as `boxed` and fail loudly otherwise. Even after the domain fix, ~5% of `boxed`-region references are still prose junk (`"**"`, `"** C. 28"`), so keep the sanity gate.

### B5 — Every rollout is truncated mid-`<think>` and graded on an abandoned draft
`configs/generation/qwen3.yaml:1` (`max_new_tokens: 1024`) + `configs/thinking/qwen3.yaml:1`, `aopd/train/rollouts.py:139-156` (no finish reason), `aopd/data/answers.py:117-144` (no `</think>` boundary)

`Rollout` carries no truncation flag and `VerificationSummary` has no `truncated` bucket, so nothing downstream can tell "ran out of budget" from "unparseable". `extract_final_answer` scans the whole decoded text; `<think>`/`</think>` are `special: False` in the Qwen3 tokenizer (ids 151667/151668) so the boundary *is* present in the string and is simply never used.

**Failure, from your pilot**: all 8 responses in `outputs/actual-model-benchmark/raw/standard.jsonl` are **exactly 1024 tokens** — 8/8 hit the cap, none terminated. 7/8 contain no `</think>` and no `\boxed`. 5 → `malformed` (silently dropped by `VerifiedWrongSelector` and by `run_opd_benchmark.py:573`, with no counter distinguishing them); 2 → `wrong`, both scraped by the prose fallback from inside an unterminated think block (`"So, final answer: (3, π/2)."` at char 3367, followed by "But let me check once again"); the 1 `correct` came from a `\boxed` at char 2502, **27 characters before** `</think>`, while its actual post-think answer was truncated away. `outputs/openthoughts-profile/summary.json`: `trace_fraction_exceeding_1024 = 1.0`, median trace 7082 tokens, p95 18527. The outcome label is a length statistic, and it is biased — short-answer problems finish, hard ones never do, which is anti-correlated with exactly the difficulty band the method targets.

**Fix**: (a) record `truncated = (len(sequence) - prompt_length >= max_new_tokens) and eos not in generated` in `RolloutCollector.collect` and add a `truncated` outcome that **no** selector retains and that is counted separately; (b) `text = text.rsplit("</think>", 1)[-1]` before extraction, and return `AnswerExtraction(None, "truncated")` when `<think>` appears without a close; (c) pick a coherent regime — either `enable_thinking: false` with 512 tokens, or thinking with ≥4096. Do not run the current in-between. Note that setting `thinking.enabled: false` alone does nothing: both drivers hardcode `enable_thinking=True` (`run_opd_benchmark.py:336`, `actual_model_benchmark.py:187`). Order matters — land the truncation metric *before* the "unterminated ⇒ not `wrong`" rule, or the pilot's pool goes to zero with no diagnostic.

### B6 — Answer extraction fabricates answers from mid-trace text and destroys valid ones
`aopd/data/answers.py:117-128`, `:130-142`

Every branch of the `for match in reversed(boxed_matches)` body returns, so the loop executes exactly once — `reversed()` suggests a fallback chain that does not exist. And `text.find("{", match.end())` at `:119` searches the *entire remaining document*. Confirmed behaviours: `"\boxed{37}. ... I used \boxed to format. The solution set is \{1,2\}."` → `"1,2\"` with `status="ok"`, silently discarding the correct 37; `"\boxed 37 which equals \frac{5}{9}."` → `"5"`; `"\boxed{37}. Let me double check: \boxed{3"` (the realistic truncation case) → `malformed`, destroying a valid earlier answer. The `####` fallback is also markdown H4: `"#### Step 1 ... #### Final Answer"` → answer `"Final Answer"`, status `ok`; `"#### Step 1 / #### Step 2 / 42"` → `"Step 2"`. The prose fallback returns the whole remainder of the line via `splitlines()[0]`, unbounded.

**Fix**: `continue` instead of `return` on malformed candidates so the loop is a real fallback chain; require `text[match.end():].startswith("{")`; tighten `_BOXED_MARKER` to `r"\\(?:boxed|fbox)\s*(?=\{)"`; restrict `####` to `^\s*####\s*(\S{1,40})\s*$`; anchor the prose fallback to the last non-empty line of the post-`</think>` region and drop the bare `answer` alternative.

### B7 — Estimator name resolution silently falls through to k3, and the run record logs what you asked for, not what ran
`aopd/losses/opd.py:56-66`, `:34-41`, `:43-53`, `aopd/train/trainer.py:378`

`_estimator_from_name` is a substring matcher with an unconditional `return "k3"`, applied to the user-supplied value at `opd.py:274` on every call. Verified: `"reverse_kl"`, `"exact_reverse_kl"`, `"full_vocab"`, `"analytic"`, `"k5"`, `"kl2"`, `""` all → `k3`; `"stop"` → `topk`. The `else: raise` at `:295` is unreachable. `from_mapping` (`:40`) silently drops any key that is not a dataclass field, so `clamp: 5.0` (instead of `clamp_log_ratio`) is discarded and the 20.0 default is used. `validate()` never relates `estimator` to `direction`, and `direction` is dead config — grep finds it only at `opd.py:27`, `:44-47` and `configs/estimator/sampled_token.yaml:3`; no computation reads it. `OPDTrainer.initialize` logs `asdict(self.loss_config)` and `save_checkpoint` stores it, so `metrics.jsonl` and the checkpoint record the raw string.

**Failure**: the todo.md ablation "compare the sampled k3 path with top-k and exact full-vocabulary estimators" — writing `estimator: exact_reverse_kl` produces a byte-identical k3 run, logged as `exact_reverse_kl`. The honest conclusion drawn would be "the estimator choice does not matter".

**Fix**: exact dict lookup that raises on unknown names; `from_mapping` raises on unrecognized keys; `validate()` enforces the estimator/direction pairing; store and log a `resolved_estimator` field.

### B8 — `topk` estimator has identically zero gradient on all student mass outside the teacher's top-k
`aopd/losses/opd.py:209-218`

The candidate set is the *teacher's* top-k; the second `log_softmax` at `:217` renormalizes over that subset, which cancels the full-vocab `logsumexp` and makes the objective depend only on student logits inside the teacher's support. Measured (V=10, k=3, float64): `d loss/d student_logit` is O(1e-17) at all seven out-of-set indices, and adding +5.0 to every out-of-set logit leaves the loss bit-identical. A student with 1.8e-18 of its mass on the teacher's top-3 reports 0.0194 against a true reverse KL of 0.7688. At V=151936 with k=32 on non-adversarial pairs it recovers 98.7%/72.3%/45.3% of the exact KL at divergences 0.13/0.48/3.04 nats — systematically biased low, and blind to off-support mass. No memory is saved: `:210` materializes the full-vocab `log_softmax` before gathering.

**Fix**: delete the arm, or keep the union of teacher and student top-k plus a lumped residual bucket `1 - Σselected` per model, computed without the second `log_softmax`.

### B9 — Sampled estimators assume `y ~ π_θ`, but rollouts come from a sharpened, truncated sampler
`aopd/losses/opd.py:193-199`, `configs/generation/qwen3.yaml:3-5`, `aopd/models/common.py:139-144`

`_sampled_logprobs` evaluates `log p_θ` at temperature 1, but generation runs at `temperature 0.6, top_p 0.95, top_k 20` with `do_sample` forced True (`run_opd_benchmark.py:327-339`). Measured on 1024 real response positions: exact `KL(p‖π_T)/token = 0.3144`, `E_q[k3] = 0.2386` (**76% of the true value**), 6.6% of the KL mass sits on tokens the sampler can never emit, and the **mean nucleus size is 1.17 tokens** — the "sampled-token" estimator is essentially deterministic at the mode. A temperature-1.0/no-truncation control returns 100.0%, confirming truncation is the cause. `cos(E_pgen[∇k3], E_p[∇k3]) = 0.319`. The bias factor depends on student peakedness, which drifts during training and differs between arms, so the logged loss is not a consistent estimate of anything across steps or baselines.

**Fix**: exact full-vocab per-token reverse KL (sampler-independent), or generate training rollouts at temperature 1.0 without truncation and keep the 0.6/20/0.95 preset for evaluation only, or importance-weight by `p(y)/p_gen(y)` and log the effective sample size. Make the rollout `GenerationOptions` a distinct config group from the eval one.

### B10 — Padding is counted as generated tokens and forwarded/backpropped in full
`aopd/train/rollouts.py:146`, `scripts/run_opd_benchmark.py:358-361` and `:569-572`, `scripts/actual_model_benchmark.py:259-262`, `scripts/generate_qwen_traces.py:209-212`

`generate(num_return_sequences=K)` right-pads all K rows to a common length; the padded row is stored verbatim as `Rollout.input_ids`, and `Rollout.attention_mask` is never assigned by anything in the repo. All four accounting sites compute generated length as `len(input_ids) - prompt_length`, and `_batch_for_rollout` fabricates `torch.ones_like(input_ids)`. Measured with the real collector (K=6, `max_new_tokens=48`): 114 reported vs 88 true tokens, 1.30×. Loss values are unaffected (`opd.py:112-113` drops `labels == pad_token_id`, and Qwen3's pad 151643 ≠ eos 151645), so this is a wrong metric on the doc's efficiency axis plus wasted forward/backward. **Currently 1.0× only because every rollout hits the cap** — it becomes material the moment B5 is fixed.

**Fix**: trim trailing pads at collection time, store a real `attention_mask` on the `Rollout`, and have all four sites and both batch builders read those fields.

### B11 — Gradient accumulation divides a per-microbatch token *mean* by a constant
`aopd/losses/opd.py:234-245`, `aopd/train/trainer.py:279`

`masked_mean` divides by that microbatch's own token count; the trainer then divides by a fixed `gradient_accumulation_steps`. The accumulated gradient is `Σ_i (L_i/N_i)/G`, not `Σ_i L_i / Σ_i N_i`, so each token is weighted `1/(G·N_i)`. Reproduced against the repo's own code: with G=2, microbatch A = 4 tokens at +1 and B = 400 tokens at −1, the trainer accumulates **5.96e-08** (exact cancellation) where the token-weighted gradient is −0.9802. Latent today — all three drivers force `gas=1` — but `configs/precision/rtx4090.yaml:5` advertises 16, and this fires the moment anyone raises it, which the B14 fix requires.

**Fix**: have the loss return `(sum_of_values, token_count)`; accumulate both; normalize once before `optimizer.step()`.

### B12 — The library's only end-to-end training loop is uncallable and carries four latent bugs
`aopd/train/trainer.py:310-355`

`fit_rollout_rounds` has zero callers (grep over `aopd/`, `scripts/`, `tests/`). It is worth fixing rather than deleting, because the fixes in §5 need it. Bugs, all confirmed by execution:
- **No `batch_builder` exists in the package** (`:314`, `:352`). The three private `_batch_for_rollout` copies take a single `Rollout` and are B=1 only. Add `aopd/train/batching.py::build_token_batch` and make it the default.
- **Never calls `initialize()`** (`:330`): raises `ModelNotLoadedError: StudentModel.load() must be called before using the tokenizer` on the first rollout, because generation precedes the lazy init in `train_token_batch`.
- **`max_steps` means "rounds" at `:326` and "optimizer steps" at `:353`.** With the shipped defaults (`max_steps: 1000`, gas=16) the break is unreachable: measured 1000 collect calls, 1000 micro-steps, **62 optimizer steps, 0 checkpoints** (`checkpoint_every: 100` > 62). Split into `max_rounds` and `max_optimizer_steps`.
- **The entire selected set goes into one micro-batch** (`:352`), and `precision.micro_batch_size` is read by no code. At 150 retained rollouts × ~1100 tokens the student logits alone are ~46 GiB. Measured on this card with the real stack: B=1 peak 12.92 GiB, B=2 14.78, B=4 19.30, **B=8 OOM on a 4.63 GiB allocation** = `8*1024*151936*4`, the fp32 `log_softmax` at `opd.py:193`.
- No final flush of a partial accumulation window at `:355`.

### B13 — Two latent alignment traps that will fire on the obvious next optimizations
`aopd/train/trainer.py:241-248` — the `labels` branch does **not** shift. Supplying HF-convention `{input_ids, labels, response_mask}` of equal length is accepted silently (verified: loss 0.7765, `response_tokens` 3) and trains `logits[:, j]` against token `j`. Delete the branch or assert `labels.shape[1] == input_ids.shape[1] - 1`.

`aopd/losses/opd.py:104-105` — `response_token_mask` is `attention & (positions >= lengths)`, a purely positional offset from index 0, while its only producer `_prompt_length` (`aopd/train/rollouts.py:63-76`) returns `attention_mask.sum()`, a token *count*. Correct for right/no padding at any batch size (verified on a ragged 2-row batch), wrong for left padding — which HF requires for batched decoder-only generation, i.e. the 3.4× speedup you need. Left-padded `[PAD,PAD,p0,p1,p2,r0,r1]` with `prompt_length=3` keeps `p1,p2` as response tokens. Rename the parameter to `response_start` and take an absolute index, or compute `first_real = attention.float().argmax(1); start = first_real + prompt_lengths`.

### B14 — Reruns merge JSONL and overwrite JSON
`scripts/run_opd_benchmark.py:40` (fixed default output dir), `:100-102` vs `:105-108`, `aopd/utils/logging.py:43`

`_write_json` truncates; `_append_jsonl` and `JsonlMetricsLogger.log` append; records carry only `event` + `timestamp`, no run id. `actual_model_benchmark.py:18-19` has hardcoded module constants and no CLI. **This has already happened**: `grep -c '"event": "initialized"'` returns **5** in both `outputs/actual-model-benchmark/{standard,active}/metrics.jsonl` (61 lines each) while `results.json` describes one run. Those loss curves are an unseparable concatenation of five invocations. Fix: timestamped run dir, refuse non-empty without `--overwrite`, stamp `run_id` on every record.

### B15 — Teacher scoring forward retains an unused KV cache
`aopd/train/trainer.py:264-271`, `configs/model/teacher/qwen3_4b.yaml:12`

`use_cache: true` is pushed onto `model.config` (`common.py:409-410`) and applies to the scoring forward. Measured on the real NF4 4B at B=4/T=1024: 1.731 GiB retained with `DynamicCache` vs 1.168 without — 0.563 GiB of pure KV (148 KiB/token) held alive through `compute_opd_loss` and the student backward. Pass `use_cache=False` and `del teacher_outputs` after extracting logits. Small at B=1 (~0.15 GiB of 12.92) but free headroom for raising the micro-batch.

---

## 3. Research-validity problems

**The Active arm is baseline #3, not the method.** `VerifiedWrongSelector` (`aopd/train/selector.py:56-76`) retains `("wrong",)`; `CorrectRolloutSelector` (`:93-103`) retains `("correct",)` — the two halves of one indicator. `docs/idea.md:150` defines Active OPD as "correctness **plus** acquisition signals such as uncertainty and near-miss score" and `:146-147` defines correctness-only as baseline #3. So the headline arm and one of its own controls are the same estimator with the sign flipped. `grep -i "acquisition|uncertain|near.miss|diversity|reliab"` over `aopd/`, `scripts/`, `configs/`, `tests/` returns nothing. Worse, `idea.md:75-77` names case 3 ("wrong and far from correct reasoning") as the *bad* case, and `verified_wrong` keeps case 2 and case 3 indiscriminately — separating them **is** the contribution, and correctness alone can never do it. → §4 for the design.

**The arms get different numbers of optimizer steps, and a block literally named `fairness_checks` reports PASS.** `run_opd_benchmark.py:510` forces `gradient_accumulation_steps: 1`; `trainer.py:284-291` then steps on every micro-batch; `:586-589` loops one `train_token_batch` per retained rollout. So `optimizer_steps == retained_rollouts`, and `:573` gives standard `("correct","wrong")` vs active `("wrong",)`. `_validate_fairness` (`:708-734`) checks weight fingerprints, two response hashes, prompt count, K, and `max_new_tokens` — and nothing about optimizer steps, trained response tokens, or teacher forward tokens, all of which it already computes. Then `:812` emits `accuracy_delta_active_minus_standard` as the result and `:803` gates the run on `all_required_checks_pass`. Your pilot: standard `optimizer_steps=8` vs active `optimizer_steps=2` on byte-identical rollouts (`responses_sha256` equal). Any delta is fully explained by update count.

**Fix**: make the *controlled resource* explicit. Fix a per-round optimizer-step budget S and per-step token budget T; every arm consumes exactly S×T, sampling with replacement from its own pool when the pool is smaller. Share one LR schedule indexed by global step. Add `same_optimizer_steps`, `same_response_tokens_trained`, `same_teacher_forward_tokens` to `_validate_fairness` as hard failures. Plot four x-axes, never one: optimizer steps, response tokens backpropagated, rollouts *generated* (identical across arms — selection buys nothing here), and teacher forward tokens (the cost selection actually reduces). `idea.md`'s chosen axis, "rollouts used for training", flatters active by construction and must not appear alone.

**No learning curve is ever produced.** `docs/idea.md:117` asks for accuracy vs gradient steps. `run_opd_benchmark` evaluates exactly twice, `:532` (`pre_training`) and `:631` (`post_training`), with `checkpoint_every: 0` at `:512` and nothing in the round loop `:544-623`. `rounds_detail` records `optimizer_steps_after_round` and `retained_rollouts` — the correct x values — with no matching y. `EvaluationConfig.update_steps` exists for exactly this (`aopd/evaluation/evaluator.py:33`) and is hardcoded to 0, never populated from `TrainerState`. Two points per arm, at *different* x, cannot establish a rate. Fix: evaluate on a fixed step grid `{0, 25, 50, 100, 200, 400}` on a fixed dev split with a fixed eval seed, appending `{optimizer_steps, cum_retained_rollouts, cum_response_tokens, accuracy, n_correct, n_total}` to `learning_curve.jsonl`.

**Two incompatible definitions of "standard OPD", and the weaker one drives the paper.** `aopd/train/selector.py:84-90` (`AllRolloutSelector`, used by `actual_model_benchmark.py:203`) retains all four outcomes — which is what `idea.md:140-141` defines. `run_opd_benchmark.py:573` hardcodes `("correct","wrong")`, silently dropping `malformed`. On your pilot's mix that is 8/8 vs **3/8** — the baseline is itself 62% ablated before the comparison starts, and since `skipped` is unreachable in that driver (`:230` rejects reference-less examples), "active" reduces to exactly "standard minus the correct rollouts". `make_selector` (`selector.py:129`) has zero callers and neither driver reads `config["filtering"]`. Fix: route every arm through `make_selector`, record the resolved `retain_outcomes` in `results.json`, and decide/document whether standard OPD sees truncated traces (it should — that is the definition — but only after B5 gives it a `truncated` label to be counted under).

**Two of the four baselines cannot be run.** `RandomRolloutSelector` and `CorrectRolloutSelector` appear only in `selector.py`, `__init__.py`, `evaluator.py` and `tests/`; no script constructs either, and `configs/filtering/` contains one file. Worse, `RandomRolloutSelector`'s budget defaults to `None` (`configs/evaluation/math500.yaml:6`, `evaluator.py:32`, `evaluator.py:143`), and with `budget=None` the `rng.sample` branch at `:123` never fires — so the "random-subset" baseline is **byte-identical to `AllRolloutSelector`** (verified: same rollout ids, same order). And when a budget *is* set, `random.Random(self.seed)` is constructed *inside* `select()` (`:124`), so every round returns the same positional indices (measured: `['2:4','0:3','0:0','2:7']` on three successive rounds). The budget-matched random control is the single thing separating "these rollouts are better" from "fewer updates are better"; it does not exist. Fix: `self._rng = random.Random(seed)` in `__init__`; make `budget` mandatory and derive it per round from the active arm's retention.

**`compare_baselines` cannot compare.** `aopd/evaluation/evaluator.py:139`: three selectors run over one frozen rollout set, and `correct_problems` is computed at `:101-106` from `records`, not from `selection.selected`. Verified: standard/random/active all return `accuracy = 0.6667` and `accuracy_per_1000_rollouts = 55.5556`, differing only in `retained_rollouts`. The docstring at `:76-79` does say this is intentional, so it is an affordance hazard rather than a computation error — but it is exported as `EvaluationResult.accuracy` and named `compare_baselines`. Rename to `summarize_selection_only` or drop the accuracy field.

**Three definitions of "accuracy" in one repo, and one published artifact reports pass@8 as accuracy.** `EfficiencyMetrics.accuracy` (`aopd/evaluation/metrics.py:22`, fed by `evaluator.py:101-106`) credits a problem if *any* of K rollouts is correct — pass@K at `num_rollouts_per_prompt: 8`, `temperature: 0.6`. `scripts/actual_model_benchmark.py:243-246,291` re-implements it inline and writes `"accuracy"`. `run_opd_benchmark._evaluation_metrics` (`:418`) is genuine pass@1. `outputs/actual-model-benchmark/results.json` therefore reports `accuracy: 1.0` for both arms from **1 correct sample out of 8** — and that number is computed from the *pre-training* rollouts at `:240-246`, before any `train_token_batch` call, while `main()` at `:359` raises unless the two arms' generations are byte-identical. The pilot's headline comparison is null by construction. Rename to `pass_at_k`, add `pass_at_1 = correct_rollouts / generated_rollouts` and `maj_at_k`, and fix `accuracy_per_1000_rollouts` (`metrics.py:35`) which is just accuracy rescaled by a constant when the rollout budget is matched.

**The study cannot see the effect it is looking for.** `--eval-samples` defaults to 128 (`:50`), one stochastic sample per problem, one seed per arm (`:786`, `:794`), no CI, no test, no repeat seeds anywhere in `results.json`. Computed: two-proportion MDE at 80% power/α=.05 is **17.0pp at n=128** (p₀=.35), 8.4pp at n=500, 5.3pp at n=1319 (GSM8K); paired McNemar at n=128 needs 7.8-13.6pp depending on discordance; 1σ per arm is 4.4pp, so a 3pp gap is 0.5σ. AIME (`idea.md:109-112`) is n=30 → MDE 31-35pp and 1σ = 8.4pp = 2.5 problems; report it as raw x/30 with no claims or drop it. Use the full 500 (`--eval-samples 0`), avg@4, Wilson intervals, paired McNemar on the per-problem vectors already written to `eval-pre.jsonl`/`eval-post.jsonl`, and ≥3 training seeds. **Do not switch eval to greedy** — the Qwen3 model card explicitly warns against greedy decoding in thinking mode; increase samples instead.

**Rollouts go stale within a round, asymmetrically.** Rollouts are collected once at `:554` and replayed as N sequential batch-size-1 optimizer steps. I measured the drift on real models (Qwen3-1.7B + NF4 4B, 24 rollouts, KL on the 1152 collected probe positions): `KL(θ₀‖teacher) = 0.3063`; after 1 step the policy has moved 0.0370 (**12.1% of the entire gap the loss exists to close**), 4 steps → 28.5%, 16 → 46.8%, 24 → 56.8%. A shipped round retains 64-96 rollouts. There is no importance weighting, no ratio clipping, no behaviour log-prob stored on `Rollout`. Fix: cap optimizer steps per generation batch at a constant M identical across arms (M=1 is the honest reference point), or store the behaviour log-prob at generation time and weight by a clipped `exp(log π_θ − log π_behaviour)`; add `optimizer_steps_per_generation` to the fairness checks.

**Adverse selection against the teacher.** Selecting student-wrong conditions on difficulty, which conditions on teacher-wrong. In a latent-difficulty simulation (b~N(0,1), P(correct)=Φ(θ−b), conditionally independent) calibrated to student .55 / teacher .75: teacher error is 25.0% overall but **39.7% on the `verified_wrong` selection** (1.59× lift) and 12.9% on the discarded set; the lift *grows* to 1.78× as the teacher improves. This is a modelling argument, not a measurement of your system — so measure it: one offline pass sampling the teacher K_T=4 per training problem gives `teacher_pass_rate`, which (a) gates the pool at ≥0.5 and (b) is a publishable table regardless of whether Active OPD wins. `todo.md:6-7` already lists this; it should be the *first* thing implemented, not the last.

**The model pair may have no headroom.** Qwen3-1.7B and Qwen3-4B in thinking mode are both in the low-to-high 90s on MATH-500, so the teacher-student gap on the primary eval is a few points — smaller than the confounds above — and the teacher is additionally NF4 while the student is bf16. (NF4 noise itself is measured at `E|t_nf4 − t_bf16| = 0.103` nats against `E|r| = 0.196`, 7.3% sign flips — but the flipped tokens have median `|r_bf16| = 0.0001` and carry only 2.35% of the gradient mass, so this is a calibration gap to measure, not a blocker.) You need ≥15-20 points of teacher advantage for the distillation gain to exceed seed noise; see §5.

---

## 4. Filtering vs. scoring — the direct answer

**Short answer: no, it does not follow, and in this codebase specifically it is likely to fail for a reason that has nothing to do with the acquisition signal.** The intuition holds in exactly one narrow sense and fails in four concrete ones.

### What each variant actually optimises

Standard OPD minimises

```
L(θ) = E_{x~D} E_{y~π_θ} [ Σ_t KL(π_θ(·|x,y_<t) ‖ π_T(·|x,y_<t)) ]
```

Any selection rule with per-rollout weight `w(x,y) ≥ 0` changes this to

```
L_w(θ) = E_{x, y~π_θ} [ w(x,y) · Σ_t kl_t ] / Z ,   Z = E[w]
```

i.e. **the same per-token KL under a tilted state distribution** `q(x,y) ∝ π_θ(y|x)·w(x,y)`. A hard filter is `w = 1{sel}`: `q` is π_θ *restricted* to a sub-support. A score is `w ∈ [0,∞)`: `q` is π_θ *tilted* but full-support.

If the student could exactly represent the teacher, every non-negative `w` shares the same minimiser (KL ≡ 0) and filtering ≡ scoring ≡ no selection, differing only in convergence rate. A 1.7B student cannot represent a 4B teacher, so under misspecification **different `w` have genuinely different optima**, not merely different rates. `docs/idea.md:156`'s claim that all baselines optimise "the same OPD objective" is false for any non-uniform `w` — filter or score.

**Where the intuition holds:** the hypothesis class of scores strictly contains filters (a step-function score reproduces any filter). So *if you construct the score so that thresholding recovers the filter exactly*, the score's family cannot be worse than the filter at its own optimum in the mixing coefficients β. That is the only sense in which "filtering works ⇒ scoring works", and it is a statement about the search space, not about what you will find in it.

### Where it fails

1. **The observed filter gain may be a count effect, not a selection effect.** In your code the filter changes the *number of optimizer steps* (`run_opd_benchmark.py:510` + `:586`) and hence data staleness (measured above: 12% of the entire student-teacher gap per step). A soft weight over all rollouts changes neither — it keeps the count and only reweights. If the filter's measured advantage came from taking fewer, fresher updates, converting it to weights reproduces **none** of it, and can invert the sign because the score restores the rollouts the filter removed. This is the dominant risk here and it is entirely an artifact of the harness, not of the idea.

2. **Ranking is a strictly harder statistical problem than thresholding.** `wrong` is one bit, near-zero variance (once B3 is fixed). Student uncertainty over K rollouts has sd `√(p(1−p)/K)` = **0.250 / 0.177 / 0.125 at K = 4 / 8 / 16** (p=0.5) — larger than most between-problem differences you want to rank on. A filter only needs the bit correct on average; a score needs the *ordering* correct. An uncertainty-weighted score can rank worse than a coin flip while the hard filter still works fine.

3. **Scale drift makes a fixed threshold non-stationary.** Correctness ∈ {0,1}, disagreement ∈ [0,1], near-miss in edit distance, teacher log-likelihood in nats and strongly length-dependent. A raw sum is dominated by whichever signal has the largest spread, and that ordering *drifts* as the student's entropy collapses during training. With a fixed absolute threshold, retention decays monotonically across rounds, silently shrinking the effective batch and effective LR — and the resulting plateau reads as "active selection stops helping". Use per-round rank- or z-normalisation and a **fixed budget m** (top-m), never a fixed threshold.

4. **Weights silently change the effective learning rate.** With unnormalised `w`, gradient magnitude ∝ Σw and noise ∝ √(Σw²); ESS = (Σw)²/Σw². (One reviewer's claim of `Σw = 15.60, ESS = 2.86` is arithmetically impossible for `w ∈ [0,1]`, where `w² ≤ w` forces `ESS ≥ Σw` — the real hazard is `ESS ≪ n`, i.e. a peaked score turning a 16-rollout batch into a 3-rollout one.) Self-normalise. And because `masked_mean` normalises by token count (`opd.py:239-245`), the correct normaliser for weighted OPD is `Σ_i w_i·len_i`, not `Σ_i w_i`.

5. **The acquisition signals are non-stationary in a way correctness is not.** Correctness is a label — stationary given the rollouts. Uncertainty and near-miss are functions of θ, which the update then changes: the states you select become the states you no longer need. Regenerating rollouts every round is the right structure, but it means seed-to-seed variance for the active arm is strictly larger than for standard OPD, so a single-seed comparison is uninformative even if everything else is fixed.

**And note the antecedent is currently unestablished.** "If the filtering criteria works well" has not been shown — the only pilot's filter output was 100% verifier false negatives (B3), and the two arms differed by 4× in update count (§3). Establish the antecedent with a budget-matched random control before spending anything on scoring.

### Minimal correct scoring design (filter recoverable as a special case)

Five changes, each small, in this order:

1. `SelectionResult` (`selector.py:31-41`) gains `weights: tuple[float, ...]`; every existing selector returns all-ones. No behaviour change; the interface exists.
2. Fix `compute_opd_loss` so it can express a weight at all. Today `opd.py:271` does `response_mask.to(dtype=torch.bool)` and `masked_mean` casts back to float — verified that `response_mask = 0.25·ones` gives a **bit-identical loss and bit-identical gradients** to `ones`. Keep `response_mask` boolean (add `if mask.is_floating_point(): raise`) and add an explicit parameter:

```python
def compute_opd_loss(..., response_mask, weights=None, config=None):
    mask = response_mask.to(dtype=torch.bool)
    w = mask.to(values.dtype)
    if weights is not None:
        w = w * weights.to(values.dtype).reshape(-1, *([1] * (w.ndim - 1)))
    return (values * w).sum() / w.sum().clamp_min(1e-6)   # self-normalised over the window
```
   Normalise over the whole accumulation window, not the micro-batch — this also fixes B11.
3. `ScoredSelector(signals, betas, budget)` computing `a_i = 1{outcome ∈ retain} · σ(Σ_k β_k · z_k(i))` with `z_k` per-round z-scored (or rank-normalised). **`β = 0` gives `a_i ≡ 0.5·1{wrong}`, which after self-normalisation is exactly `VerifiedWrongSelector`.** Assert that in a test — that is what makes the comparison controlled: the only thing that differs between the filter arm and the score arm is β.
4. Select **top-m** with `m = |verified_wrong retention|` for that round, so the token and step budget match the filter arm by construction.
5. Signals, all nearly free from what already exists:
   - **Group disagreement** `u(x) = 1 − max_a freq_a` over the K rollouts of a prompt. `metadata["prompt_index"]`/`num_rollouts_per_prompt` are already populated at `rollouts.py:150-154`; no selector groups by them today. Use it as a *problem-level* weight (averaged over K) to cut the 0.177 sd, not a rollout-level one.
   - **Near-miss** = normalised distance from the rollout's answer/trace to the group's correct or modal rollout, **gated on `1{group contains ≥1 correct rollout}`**. This is precisely the case-2 vs case-3 discriminator `idea.md:75-77` asks for and the one thing correctness alone can never provide.
   - **Teacher reliability** = mean teacher log-prob per response token of the student trace — already computed as `teacher_selected` at `opd.py:198` and thrown away; cache it onto the `Rollout`. Plus `teacher_pass_rate` from the one-off offline pass. **Fix the sign before you run**: high teacher likelihood ⇒ teacher agrees with the student's path ⇒ near-zero KL ⇒ near-zero gradient; low ⇒ large KL but an off-manifold target. `idea.md` names "reliability" without fixing the sign, and the two choices give opposite methods.
   - **Diversity** via greedy facility-location over normalised answers within a round.

**Ablation ladder**, ordered so each rung answers one question and can be cut when the budget runs out:
(i) matched-budget random vs `verified_wrong` at fixed count — does *which* rollout matter at all? **If this rung is null, stop; nothing downstream can be true.**
(ii) add the teacher-reliability gate — does removing teacher-wrong problems help?
(iii) uncertainty as a problem-level weight.
(iv) near-miss + teacher likelihood, sign pre-registered.

---

## 5. Compute-feasible plan for one RTX 4090

Measured constants to budget against: sequential collection **16.1 s/prompt** (K=8, 1024 tokens, thinking) vs **4.8 s/prompt** left-padded batched at bs=4 — 3.37× — so `RolloutCollector.collect`'s one-`generate`-per-prompt loop (`rollouts.py:121-132`, with `encode_prompt` defaulting `padding=False` at `common.py:299`) must be batched before anything else; generation is ~87% of wall clock (17.7 s generation vs 2.6 s training in your recorded run). Loss cost 1.74 MiB/token; full stack peaks 12.92 GiB at B=1/T=1024 and OOMs at B=8. One 500-problem MATH-500 eval point: 134 min sequential, 40 min batched.

**Rung 0 — zero GPU, hours of work.** Land B1 (Kahan/stochastic-rounding optimizer + the "did params move" assertion), B2 (route to `compute_policy_gradient_opd_loss` + the expected-gradient test), B3 (`math_verify` + `unverified` outcome), B4 (math corpus with a real answer column), B5 (truncation flag + `</think>` gating), B6, B7. Then **re-label `outputs/actual-model-benchmark/raw/*.jsonl` offline** with the fixed verifier and extractor and report the old gate's false-negative rate. That table is the cheapest real result in this project and it costs no GPU.

**Rung 1 — smoke, ~15 min GPU.** 4 prompts, K=4, new token budget. Assert: >50% of sampled student params change after 8 steps; reverse KL on the collected probe positions decreases; truncation rate <20%; batched collection reproduces the sequential rollouts' verifier outcomes.

**Rung 2 — the gate, ~2 GPU-hours.** Offline teacher pass: K_T=4 on 512 math problems, record `teacher_pass_rate`. Report teacher accuracy on `verified_wrong`-selected vs discarded problems. If the adverse-selection lift is near 1.0 the method is on safer ground than the simulation suggests; if it is ~1.6 you must gate on `teacher_pass_rate ≥ 0.5` before anything else, and the table is itself a contribution.

**Rung 3 — the actual experiment, ~1.5 GPU-days.**
- **Models**: student Qwen3-0.6B bf16, teacher Qwen3-8B NF4. Memory: 1.2 (weights) + 1.2 (grads) + ~1.2 (8-bit Adam) + ~5.2 (NF4 8B) ≈ 8.8 GiB, leaving ~14 GiB for logits — B=4×T=1024 at 1.74 MiB/token is ~7 GiB, comfortable, and B=8 with the chunked loss. This buys a ≥15-20pt capability gap; 1.7B→4B on MATH-500 does not, and at a 1024-token thinking budget neither model reaches its published score anyway.
- **Regime**: non-thinking, `max_new_tokens: 512`. (If you insist on thinking, ≥4096 and treat `truncated` as a first-class always-retained outcome for the standard arm.)
- **Data**: MATH train levels 3-5, 2k problems. K=4, 64 prompts/round, 40 rounds. Fixed per-round budget: S optimizer steps × T tokens, identical across arms.
- **Arms**: `all` / `random-matched-to-active` / `verified_wrong` / `scored(β)`. 3 seeds.
- **Eval**: 200-problem dev split avg@4 at 6 checkpoints for the curve; full MATH-500 + GSM8K-test avg@4 at the endpoints only. Fixed eval seed across all phases and arms.
- **Cost**: generation 64×4×512 = 131k tokens/round; at ~2-4k tok/s batched for a 0.6B that is 35-65 s/round, ~30-45 min per arm-seed over 40 rounds. Curve evals 6 × 200 × 4 × 512 ≈ 2.5M tokens ≈ 15-25 min. Training ≈ 5 min. Call it ~1.5 h per arm-seed × 12 = ~18 GPU-hours, plus ~4 h of endpoint evals. If that is too long, drop to 2 seeds and 3 arms (kill `correct-only`).
- Serve generation from vLLM refreshed once per round if you can spare the integration time; it is a 3-5× multiplier on the 87% of the budget that is sampling.

**Rung 4** — only if Rung 3's rung (i) (matched-budget random vs `verified_wrong`) is non-null: add the acquisition signals one at a time per §4's ladder.

State the honest framing on the plots: all arms generate identically, so **selection saves training compute, not sampling compute**. That is a real and defensible claim; "rollouts used for training" alone is not.

---

## 6. Lower-priority cleanups

- `masked_mean` (`opd.py:238`) does `(values * mask).sum()`, so `inf * 0 = NaN` at a masked position poisons the batch. Use `torch.where(mask, values, zeros)`. Reachable only with `clamp_log_ratio: null`, which no shipped config sets.
- `clamp_log_ratio: 20.0` (`opd.py:30`) guards the branch that cannot occur. Measured sampled-`r` quantiles on real rollouts: `[-2.10, -1.22, -0.45, 0.00, 4.60, 15.85, 20.37]` — `r < -20` is structurally impossible for a token the student itself sampled, while `r > +20` **did** bind (1 token in 1024) and gets exactly zero gradient there, i.e. the dead zone is on the tokens the teacher most rejects. Pick the threshold from the measured range (~2-20) and cap the gradient weight, not its logarithm; log the clamped fraction per step.
- `estimator.response_tokens_only` is declared on `OPDLossConfig` and never consulted by `compute_opd_loss`. Delete or implement.
- Autocast: `_configure_amp` (`trainer.py:434`) infers the dtype from the model *load* dtype. The fp32 promotion of `log_softmax` this causes is a numerical **benefit** (relative error 4.5e-3 with autocast vs 3.8e-2 without at V=151936), not overhead — do not disable it; the fix is the chunked per-token log-prob, which makes the dtype of a `[B,T,V]` tensor irrelevant. (Both drivers also assert `amp_enabled and active_amp_dtype == "bfloat16"` at `run_opd_benchmark.py:161-165` / `actual_model_benchmark.py:72-76`, so disabling it would hard-fail them.)
- Dead Hydra surface: `configs/filtering/*` and `configs/rollout/*` are read by no driver at all (`grep 'config\["..."\]'` over both scripts); `require_exact_match`, `log_outcomes`, `lazy_dataset_loading`, `micro_batch_size`, `activation_checkpointing`, `peak_memory_fraction`, `run_dir`, `checkpoint_dir`, `log_dir`, `seed_value`, `optimizer.name`, `thinking.add_generation_prompt` have zero readers. Note `run_opd_benchmark` uses argparse, not the Hydra CLI, so `filtering.policy=random` on the command line **errors out** rather than being silently ignored — the failure is loud there, and only silent for someone constructing configs programmatically. Wire the groups or delete the keys; make `from_mapping` raise on unknown keys.
- `configs/precision/rtx4090.yaml:5` advertises `gradient_accumulation_steps: 16` but all three drivers force 1 (`run_opd_benchmark.py:510`, `actual_model_benchmark.py:158`, `real_model_smoke.py:106`), and there is no LR warmup or scheduler anywhere (`_build_optimizer` returns a bare AdamW; `_clip_and_step` only clips and steps). The realised regime is 16× more updates than documented at the LR chosen for the documented one, from step 0. Record the effective batch size in `results.json`.
- `RolloutCollector._generation_options` (`rollouts.py:172-175`) silently overrides caller-supplied `max_new_tokens` and `num_return_sequences`. Currently harmless (every driver passes matching values), but raise on conflict.
- `retain_prompt_tokens: false` (`rollouts.py:146`) stores `input_ids=None` while still populating `prompt_length`, and training then dies with `RuntimeError: Selected rollout did not retain tokenized input IDs` *after* the full generation cost. Rename to `retain_token_ids` and validate at trainer construction.
- `rollout_id = f"{example_index}:{rollout_index}"` (`rollouts.py:144`) is unique only within one `collect()` call — `outputs/actual-model-benchmark/raw/{standard,active}.jsonl` both contain `['0:0'..'0:7']`. Add the round index and `optimizer_steps_at_collection` to the id and metadata; the latter is the staleness bookkeeping §3 needs.
- No `load_checkpoint`/resume exists anywhere (grep), and `fit_rollout_rounds` never writes a final checkpoint. The storage-exhaustion scenario one reviewer raised is refuted — all three drivers set `checkpoint_every: 0` — but a run that OOMs mid-way cannot be resumed.
- `summarize_verification(rollouts, reference_answer=...)` broadcasts one reference onto every rollout and **permanently overwrites** `self.reference_answer` (`rollouts.py:34-35`, `:87`). No caller passes it today; make `Rollout.verify` non-mutating.
- `pyproject.toml:14`: `datasets` is under the `data` extra but every entry point except `main.py` needs it (the `ImportError` fires at `run_opd_benchmark.py:761`, *after* the CUDA preflight); `numpy` is undeclared; `transformers>=4.51` has no upper bound and the installed 5.14.1 already warns `torch_dtype is deprecated` on the `pretrained_kwargs` path (`common.py:70`) and silently discards `low_cpu_mem_usage` (`:71`). The force-included top-level `configs/` package will collide in site-packages. Note `use_8bit: true` does *not* require bitsandbytes — `trainer.py:190-202` catches the ImportError and falls back.
- Test suite: 24/24 pass while these mutants survive — `AllRolloutSelector` retaining `("correct","wrong")`; `CorrectRolloutSelector` retaining `("wrong",)`; `labels = input_ids[:, :-1]`; dropping `prompt_lengths - 1`; `attention_mask[:, :-1]`; `rng.sample → selected[:budget]`; `log_ratio.detach() + 0.0*log_ratio`. Add: the expected-gradient test from B2; a hand-shifted ragged padded multi-row equivalence test for `train_token_batch`; a table-driven test per selection policy asserting the exact retained set; the verifier regression table from B3; a left-padded `response_token_mask` case. Also, `tests/test_synthetic_smoke.py:60` passes an explicit optimizer, so `TrainerConfig(weight_decay=0.0)` is shadowed by torch's default 0.01 — verified that with a zeroed OPD gradient the "loss decreases" assertion still passes on weight decay alone. And both AMP tests (`tests/test_trainer_amp.py:37`) copy the student state dict into the teacher unperturbed, so they assert `isfinite(0.0)`; the CUDA autocast branches (`trainer.py:457-458`, `466-469`, `545-548`) are unexecuted even on this GPU box.
- `aopd/models/common.py` is at 31% line coverage and `aopd/train/rollouts.py` at 25% — `generate`, `prepare_inputs`, `encode_prompt`, `assert_tokenizer_compatible`, `to_generate_kwargs`, `RolloutCollector.collect` and `_prompt_length` are all untested, and `_prompt_length` is what produces the `prompt_lengths` the entire loss mask depends on. Drive a `FakeTokenizer`/`FakeCausalLM` through the real `CausalLMWrapper` rather than the bespoke `TinyWrapper` stub.
- **Non-issue, checked so you don't have to**: train/eval contamination. I scanned all 113,957 OpenThoughts user turns against all 500 MATH-500 problems (case/punctuation-normalised): **0 exact matches**; the 82 13-gram-flagged pairs are all template siblings or shared `asy` boilerplate. Add a decontamination hash pass for provenance, but it is not a confound for this dataset pair. (`example_from_record`'s id tuple at `datasets.py:82` does miss MATH-500's `unique_id` column, while `scripts/openthoughts_utils.py:59` includes it — worth unifying so eval items are addressable.)