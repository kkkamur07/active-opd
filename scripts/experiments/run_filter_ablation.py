"""Budget-matched filtered-vs-unfiltered OPD ablation.

This is the kill-switch experiment: does it matter *which* rollouts are trained
on, once every arm gets the same number of optimizer steps and the same number
of backpropagated response tokens?

The previous benchmark could not answer that. It gave the "standard" arm
``("correct", "wrong")`` and the "active" arm ``("wrong",)`` with one optimizer
step per retained rollout, so the arms differed by 4x in update count (8 vs 2 in
the committed pilot) while a block named ``fairness_checks`` reported PASS --
it validated the *generation* budget and never the *training* budget.

Arms
----
``all``              every rollout the verifier decided (standard OPD)
``random``           a random subset matched to the active arm's count
``verified_wrong``   the Active OPD gate

All three see byte-identical rollouts each round (one generation pass, shared),
so any difference is attributable to selection alone. Each arm is then matched
to a common per-round budget of ``--select-budget`` rollouts.

Usage
-----
    python -m scripts.run_filter_ablation --train-prompts 64 --rounds 20 \
        --k 4 --select-budget 16 --eval-problems 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARMS = ("all", "random", "verified_wrong")


@dataclass(frozen=True)
class AblationOptions:
    train_prompts: int = 64
    rounds: int = 20
    k: int = 4
    select_budget: int = 16
    eval_problems: int = 200
    eval_samples: int = 4
    max_new_tokens: int = 18000
    seed: int = 0
    #: Fixed across arms and phases so learning-curve points are comparable.
    eval_seed: int = 1234
    dataset_preset: str = "openr1"
    student_model: str | None = None
    teacher_model: str | None = None
    output_dir: str = "outputs/filter-ablation"
    eval_every: int = 5
    arms: tuple[str, ...] = ARMS
    dry_run: bool = False


def _parse_args(argv: list[str] | None = None) -> AblationOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-prompts", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--k", type=int, default=4, help="rollouts per prompt")
    parser.add_argument(
        "--select-budget",
        type=int,
        default=16,
        help="rollouts trained on per round, identical for every arm",
    )
    parser.add_argument("--eval-problems", type=int, default=200)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=18000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-preset", default="openr1")
    parser.add_argument("--student-model", default=None)
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--output-dir", default="outputs/filter-ablation")
    parser.add_argument("--eval-every", type=int, default=5, help="rounds between evals")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate wiring and print the plan without loading models",
    )
    args = parser.parse_args(argv)
    return AblationOptions(
        train_prompts=args.train_prompts,
        rounds=args.rounds,
        k=args.k,
        select_budget=args.select_budget,
        eval_problems=args.eval_problems,
        eval_samples=args.eval_samples,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        dataset_preset=args.dataset_preset,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        output_dir=args.output_dir,
        eval_every=args.eval_every,
        arms=tuple(args.arms),
        dry_run=args.dry_run,
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _load_examples(options: AblationOptions) -> tuple[list[Any], list[Any]]:
    from aopd.data.datasets import iter_examples, load_math500, load_math_training_set

    records, example_kwargs = load_math_training_set(
        options.dataset_preset, seed=options.seed
    )
    train = list(
        iter_examples(
            records,
            limit=options.train_prompts,
            skip_invalid=True,
            **example_kwargs,
        )
    )
    if len(train) < options.train_prompts:
        raise RuntimeError(
            f"Only {len(train)} training examples had a usable ground-truth answer; "
            f"{options.train_prompts} requested."
        )
    evaluation = list(
        iter_examples(
            load_math500(),
            limit=options.eval_problems,
            skip_invalid=True,
        )
    )
    return train, evaluation


def _build_models(options: AblationOptions):
    from aopd.models import StudentModel, TeacherModel

    student_config = {"model_id": options.student_model} if options.student_model else None
    teacher_config = {"model_id": options.teacher_model} if options.teacher_model else None
    student = StudentModel(student_config) if student_config else StudentModel()
    teacher = TeacherModel(teacher_config) if teacher_config else TeacherModel()
    return student, teacher


def _evaluate(
    student: Any,
    examples: list[Any],
    options: AblationOptions,
    *,
    phase: str,
    optimizer_steps: int,
    response_tokens: int,
) -> dict[str, Any]:
    from aopd.evaluation import evaluate_rollout_groups
    from aopd.models import GenerationOptions
    from aopd.train import RolloutCollector
    from aopd.utils.reproducibility import seed_everything

    # A fixed eval seed across arms and phases so curve points are comparable.
    seed_everything(options.eval_seed)
    collector = RolloutCollector(
        student,
        config={
            "num_rollouts_per_prompt": options.eval_samples,
            "max_new_tokens": options.max_new_tokens,
            "prompt_batch_size": 4,
        },
    )
    generation = GenerationOptions(
        max_new_tokens=options.max_new_tokens,
        num_return_sequences=options.eval_samples,
        do_sample=True,
    )
    started = time.perf_counter()
    rollouts = collector.collect(examples, generation=generation)
    groups: dict[str, list[Any]] = {}
    for rollout in rollouts:
        groups.setdefault(str(rollout.metadata.get("prompt_index")), []).append(rollout)
    result = evaluate_rollout_groups(
        groups.values(),
        phase=phase,
        elapsed_seconds=time.perf_counter() - started,
        optimizer_steps=optimizer_steps,
        response_tokens_trained=response_tokens,
    )
    return result.as_dict()


def _run_arm(
    arm: str,
    options: AblationOptions,
    train_examples: list[Any],
    eval_examples: list[Any],
    run_dir: Path,
) -> dict[str, Any]:
    from aopd.data.rollouts import ALL_OUTCOMES
    from aopd.models import GenerationOptions
    from aopd.train import (
        OPDTrainer,
        RolloutCollector,
        TrainerConfig,
        make_selector,
        match_budget,
    )
    from aopd.losses import OPDLossConfig
    from aopd.utils.reproducibility import seed_everything

    arm_dir = run_dir / arm
    student, teacher = _build_models(options)

    trainer = OPDTrainer(
        student,
        teacher,
        config=TrainerConfig(
            max_optimizer_steps=10**9,
            max_rounds=options.rounds,
            gradient_accumulation_steps=1,
            micro_batch_size=1,
            output_dir=str(arm_dir),
            seed=options.seed,
            master_weights="fp32",
            use_8bit_optimizer=False,
            warmup_steps=10,
            checkpoint_every=0,
            assert_params_move_after=4,
        ),
        loss_config=OPDLossConfig(estimator="exact_reverse_kl"),
    ).initialize()

    collector = RolloutCollector(
        student,
        config={
            "num_rollouts_per_prompt": options.k,
            "max_new_tokens": options.max_new_tokens,
            "prompt_batch_size": 4,
        },
    )
    generation = GenerationOptions(
        max_new_tokens=options.max_new_tokens,
        num_return_sequences=options.k,
        do_sample=True,
    )
    selector = make_selector(
        arm, budget=options.select_budget, seed=options.seed + 17
    )
    build_batch = trainer._default_batch_builder()

    curve_path = arm_dir / "learning_curve.jsonl"
    outcomes: Counter[str] = Counter()
    rounds_detail: list[dict[str, Any]] = []

    baseline = _evaluate(
        student,
        eval_examples,
        options,
        phase="step-0",
        optimizer_steps=0,
        response_tokens=0,
    )
    _append_jsonl(curve_path, {"arm": arm, **baseline})

    for round_index in range(options.rounds):
        # Rollouts are regenerated every round with the current weights, so the
        # states stay on-policy.
        seed_everything(options.seed + 1000 + round_index)
        rollouts = collector.collect(
            train_examples, generation=generation, round_index=round_index
        )
        selection = selector.select(rollouts)
        # Match every arm to the same number of trained rollouts, so the
        # comparison is about *which* rollouts, not how many updates.
        matched = match_budget(
            selection, options.select_budget, seed=options.seed + round_index
        )
        for outcome in ALL_OUTCOMES:
            outcomes[outcome] += selection.summary.count(outcome)

        for rollout in matched.selected:
            trainer.train_token_batch(build_batch([rollout]))

        rounds_detail.append(
            {
                "round": round_index,
                "generated_rollouts": selection.summary.total,
                "pool_size": len(selection.selected),
                "trained_rollouts": len(matched.selected),
                "usable_rate": selection.summary.usable_rate,
                "retention_rate": selection.retained_rollout_rate,
                "outcomes": dict(selection.summary.counts),
                "optimizer_steps": trainer.state.optimizer_steps,
                "response_tokens_trained": trainer.state.response_tokens_trained,
            }
        )
        _append_jsonl(arm_dir / "rounds.jsonl", {"arm": arm, **rounds_detail[-1]})

        is_last = round_index == options.rounds - 1
        if is_last or (round_index + 1) % options.eval_every == 0:
            point = _evaluate(
                student,
                eval_examples,
                options,
                phase=f"round-{round_index + 1}",
                optimizer_steps=trainer.state.optimizer_steps,
                response_tokens=trainer.state.response_tokens_trained,
            )
            _append_jsonl(curve_path, {"arm": arm, **point})

    trainer.flush_gradients()
    for wrapper in (student, teacher):
        unload = getattr(wrapper, "unload", None)
        if callable(unload):
            unload()

    return {
        "arm": arm,
        "state": asdict(trainer.state),
        "verifier_outcomes": dict(outcomes),
        "rounds_detail": rounds_detail,
        "learning_curve": str(curve_path),
    }


def validate_fairness(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Check the budgets that actually confound a learning-efficiency claim.

    The previous version checked prompt count, K and generation budget -- the
    generation side -- and nothing about optimizer steps or trained tokens.
    """

    states = {arm: result["state"] for arm, result in results.items()}
    steps = {arm: state["optimizer_steps"] for arm, state in states.items()}
    tokens = {arm: state["response_tokens_trained"] for arm, state in states.items()}
    generated = {arm: state["generated_rollouts"] for arm, state in states.items()}

    def _spread(values: dict[str, int]) -> float:
        if not values or max(values.values()) == 0:
            return 0.0
        return (max(values.values()) - min(values.values())) / max(values.values())

    checks = {
        "same_optimizer_steps": len(set(steps.values())) == 1,
        "same_rollouts_generated": len(set(generated.values())) == 1,
        # Token counts cannot match exactly (rollouts have different lengths);
        # require them within 10% so no arm trains on materially more text.
        "response_tokens_within_10pct": _spread(tokens) <= 0.10,
    }
    checks["all_required_checks_pass"] = all(checks.values())
    return {
        "checks": checks,
        "optimizer_steps": steps,
        "response_tokens_trained": tokens,
        "generated_rollouts": generated,
        "response_token_spread": _spread(tokens),
    }


def _plan(options: AblationOptions) -> dict[str, Any]:
    per_round = options.train_prompts * options.k
    return {
        "rollouts_generated_per_round_per_arm": per_round,
        "rollouts_trained_per_round_per_arm": options.select_budget,
        "optimizer_steps_per_arm": options.select_budget * options.rounds,
        "total_rollouts_generated": per_round * options.rounds * len(options.arms),
        "eval_points_per_arm": options.rounds // max(options.eval_every, 1) + 1,
        "eval_generations_per_point": options.eval_problems * options.eval_samples,
        "arms": list(options.arms),
    }


def main(argv: list[str] | None = None) -> int:
    options = _parse_args(argv)
    run_dir = Path(options.output_dir) / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    plan = _plan(options)

    base: dict[str, Any] = {
        "status": "starting",
        "started_at": datetime.now(UTC).isoformat(),
        "options": asdict(options),
        "plan": plan,
        "command": " ".join(sys.argv),
    }
    _write_json(run_dir / "results.json", base)
    print(json.dumps({"run_dir": str(run_dir), "plan": plan}, indent=2))

    if options.dry_run:
        base["status"] = "dry-run"
        _write_json(run_dir / "results.json", base)
        return 0

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing to run on CPU.")
        from aopd.utils.reproducibility import configure_cuda_memory

        configure_cuda_memory(allow_tf32=True)

        train_examples, eval_examples = _load_examples(options)
        results: dict[str, dict[str, Any]] = {}
        for arm in options.arms:
            print(f"[filter-ablation] running arm: {arm}", flush=True)
            results[arm] = _run_arm(arm, options, train_examples, eval_examples, run_dir)

        base.update(
            {
                "status": "ok",
                "finished_at": datetime.now(UTC).isoformat(),
                "arms": results,
                "fairness": validate_fairness(results),
                "train_examples": len(train_examples),
                "eval_problems": len(eval_examples),
            }
        )
        _write_json(run_dir / "results.json", base)
        fairness = base["fairness"]["checks"]
        print(json.dumps(base["fairness"], indent=2))
        if not fairness["all_required_checks_pass"]:
            print(
                "FAIRNESS CHECKS FAILED: the arms did not consume matched budgets, "
                "so any accuracy delta is confounded.",
                file=sys.stderr,
            )
            return 3
        return 0
    except Exception as exc:  # noqa: BLE001 - report and persist the failure
        base.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "failed_at": datetime.now(UTC).isoformat(),
            }
        )
        _write_json(run_dir / "results.json", base)
        print(f"FILTER_ABLATION_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
