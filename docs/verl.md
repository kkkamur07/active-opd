# Running the experiment on verl

verl ships official on-policy distillation support, which matters here because
generation is roughly 87 percent of wall clock in the HF trainer path and verl
serves rollouts through vLLM. On a 4x A100 80GB node the natural split is two
GPUs for the student trainer and two for the teacher pool.

Docs: <https://verl.readthedocs.io/en/latest/algo/opd.html> and the async
recipe at <https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html>.

## Minimal mapping

verl's OPD runs through `main_ppo.py` with a dedicated teacher resource pool:

```yaml
distillation:
  enabled: true
  nnodes: 1
  n_gpus_per_node: 2            # teacher pool: 2 of the 4 A100s
  teacher_models:
    teacher_model:
      model_path: Qwen/Qwen3-4B
      inference:
        name: vllm
        gpu_memory_utilization: 0.8
        max_model_len: 32768    # prompt + 30k-token thinking trace
  distillation_loss:
    loss_mode: k2               # see the estimator warning below
    use_policy_gradient: false

actor_rollout_ref:
  actor:
    use_kl_loss: false          # the teacher KL replaces the ref-policy KL
algorithm:
  use_kl_in_reward: false
```

Set the rollout length so traces are never truncated: response length 30000
with `max_model_len: 32768`, same as `configs/generation/qwen3_thinking.yaml`.

## The estimator warning

verl's GKD-style OPD (`use_policy_gradient: false`) backpropagates directly
through the student probabilities of a sampled KL estimator. That is the exact
construction this repo removed from its own trainer: the pathwise derivative
of `k3` is the *forward* KL gradient, and the pathwise derivative of `k1` has
expectation zero. `docs/code-review.md` has the derivation.

Safe choices in verl:

- `loss_mode: k2` with `use_policy_gradient: false`. The pathwise gradient of
  `0.5 r^2` is a consistent reverse-KL gradient estimator. This is the direct
  analogue of this repo's `k2` estimator.
- `use_policy_gradient: true` with `k1` or `k3` as the value. PG OPD treats
  the negative estimator as a reward, which is the score-function form and is
  consistent regardless of which value estimator supplies the number.

Do not run `loss_mode: k3` (or `low_var_kl`, the same thing) with
`use_policy_gradient: false` and interpret the result as reverse-KL training.

`forward_kl_topk` is a different objective (mode-covering, not mode-seeking);
this repo's `topk` estimator is its reverse-KL counterpart if you want a
comparison.

## What verl does not give you

verl distills on every rollout. The selection step, which is the entire
research question here, has no hook in `distillation_loss`. Two ways to keep
the ablation honest under verl:

1. Filter between rollout and update. verl's agent loop yields the batch
   before the distillation update; drop or reweight samples there using
   `aopd.data.answers.verify_exact_answer` and `aopd.train.selector`. The
   budget-matching rules from `scripts/experiments/run_filter_ablation.py`
   still apply: arms must see identical rollouts and matched trained tokens,
   or the comparison is unreadable.
2. Use verl only as a generation service. Keep this repo's trainer and
   verifier, and replace `collect_rollouts`' HF `generate` calls with requests
   to a vLLM server that is refreshed with student weights each round. This
   is the smaller change and keeps every fairness check intact.

Either way, run the fairness checks. verl reports neither trained response
tokens nor per-arm budget parity.

## Version note

Written against the verl docs as of August 2026 (post v0.7). The
`distillation.*` keys are recent; pin the verl version you validate against.
