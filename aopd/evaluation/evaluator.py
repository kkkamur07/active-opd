"""Held-out evaluation for standard and Active OPD.

Evaluation is deliberately separate from selection.  ``pass@1`` over sampled
responses is the headline metric; per-problem correctness vectors are kept so
arms can be compared with a paired test rather than two independent
proportions.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from aopd.data.datasets import MathExample
from aopd.data.rollouts import Rollout
from aopd.utils.reproducibility import peak_cuda_memory

from .metrics import EfficiencyMetrics, mcnemar_pvalue


@dataclass(frozen=True)
class EvaluationConfig:
    """Controls for a held-out evaluation pass."""

    num_problems: int | None = None
    samples_per_problem: int = 4
    eval_seed: int = 1234
    max_new_tokens: int | None = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> EvaluationConfig:
        names = {field_.name for field_ in cls.__dataclass_fields__.values()}
        return cls(**{name: config[name] for name in names if name in config})


@dataclass
class EvaluationResult:
    """Metrics plus the per-problem detail needed for a paired comparison."""

    metrics: EfficiencyMetrics
    #: problem_id -> number of correct samples for that problem.
    per_problem_correct: dict[str, int] = field(default_factory=dict)
    #: problem_id -> number of samples drawn for that problem.
    per_problem_total: dict[str, int] = field(default_factory=dict)
    phase: str = ""

    @property
    def accuracy(self) -> float:
        return self.metrics.pass_at_1

    def solved_mask(self) -> dict[str, bool]:
        """Majority-solved per problem, for paired testing between arms."""

        return {
            problem: self.per_problem_correct.get(problem, 0) * 2
            > self.per_problem_total.get(problem, 0)
            for problem in self.per_problem_total
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self.metrics.as_dict()
        payload.update(
            {
                "phase": self.phase,
                "per_problem_correct": dict(self.per_problem_correct),
                "per_problem_total": dict(self.per_problem_total),
            }
        )
        return payload


def evaluate_rollout_groups(
    groups: Iterable[Sequence[Rollout]],
    *,
    phase: str = "",
    elapsed_seconds: float | None = None,
    optimizer_steps: int = 0,
    response_tokens_trained: int = 0,
) -> EvaluationResult:
    """Score evaluation rollouts, one group per problem.

    ``pass@1`` is computed over every generated response; ``pass@k`` is
    reported separately rather than being called "accuracy".
    """

    started = time.perf_counter()
    outcomes: Counter[str] = Counter()
    correct_rollouts = 0
    generated = 0
    problems_any_correct = 0
    total_problems = 0
    per_problem_correct: dict[str, int] = {}
    per_problem_total: dict[str, int] = {}

    for index, group in enumerate(groups):
        records = list(group)
        if not records:
            continue
        total_problems += 1
        problem_id = str(
            records[0].metadata.get("problem_id") or records[0].rollout_id or index
        )
        group_correct = 0
        for rollout in records:
            result = rollout.verification or rollout.verify()
            outcomes[result.outcome] += 1
            generated += 1
            if result.outcome == "correct":
                group_correct += 1
        correct_rollouts += group_correct
        per_problem_correct[problem_id] = group_correct
        per_problem_total[problem_id] = len(records)
        if group_correct:
            problems_any_correct += 1

    duration = (
        elapsed_seconds
        if elapsed_seconds is not None
        else time.perf_counter() - started
    )
    metrics = EfficiencyMetrics(
        correct_rollouts=correct_rollouts,
        generated_rollouts=generated,
        problems_any_correct=problems_any_correct,
        total_problems=total_problems,
        optimizer_steps=optimizer_steps,
        response_tokens_trained=response_tokens_trained,
        elapsed_seconds=duration,
        peak_memory_bytes=peak_cuda_memory(),
        verifier_outcomes=dict(outcomes),
    )
    return EvaluationResult(metrics, per_problem_correct, per_problem_total, phase)


def evaluate_dataset(
    examples: Iterable[MathExample],
    rollout_fn: Callable[[MathExample], Iterable[Rollout]],
    *,
    phase: str = "",
    optimizer_steps: int = 0,
    response_tokens_trained: int = 0,
) -> EvaluationResult:
    """Generate lazily from a held-out dataset and score it."""

    started = time.perf_counter()
    groups: list[Sequence[Rollout]] = []
    for example in examples:
        records = list(rollout_fn(example))
        for rollout in records:
            if rollout.reference_answer is None:
                rollout.reference_answer = example.reference_answer
            rollout.metadata.setdefault("problem_id", example.problem_id)
        groups.append(records)
    return evaluate_rollout_groups(
        groups,
        phase=phase,
        elapsed_seconds=time.perf_counter() - started,
        optimizer_steps=optimizer_steps,
        response_tokens_trained=response_tokens_trained,
    )


def paired_comparison(
    first: EvaluationResult,
    second: EvaluationResult,
) -> dict[str, Any]:
    """Compare two arms on the problems they both saw (McNemar)."""

    left, right = first.solved_mask(), second.solved_mask()
    shared = sorted(set(left) & set(right))
    only_first = sum(1 for key in shared if left[key] and not right[key])
    only_second = sum(1 for key in shared if right[key] and not left[key])
    return {
        "shared_problems": len(shared),
        "only_first": only_first,
        "only_second": only_second,
        "accuracy_delta": first.metrics.pass_at_1 - second.metrics.pass_at_1,
        "mcnemar_p_value": mcnemar_pvalue(only_first, only_second),
    }


__all__ = [
    "EvaluationConfig",
    "EvaluationResult",
    "evaluate_dataset",
    "evaluate_rollout_groups",
    "paired_comparison",
]
