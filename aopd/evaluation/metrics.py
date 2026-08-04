"""Accuracy and learning-efficiency metrics for math evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.96,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Reported instead of a bare accuracy because the effect sizes this project
    is chasing are comparable to the sampling noise: at n=128 the minimum
    detectable difference between two arms is around 17 percentage points.
    """

    if trials <= 0:
        return (0.0, 0.0)
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    centre = proportion + z**2 / (2 * trials)
    spread = z * math.sqrt(
        proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)
    )
    return ((centre - spread) / denominator, (centre + spread) / denominator)


def mcnemar_pvalue(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts.

    Arms are evaluated on the *same* problems, so the paired test is far more
    sensitive than comparing two independent proportions.
    """

    n = only_a + only_b
    if n == 0:
        return 1.0
    smaller = min(only_a, only_b)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    return min(1.0, 2 * tail)


@dataclass
class EfficiencyMetrics:
    """Accuracy plus the resource axes a learning-efficiency claim needs.

    ``pass_at_1`` is the headline number. ``pass_at_k`` (any-of-K correct) was
    previously reported as "accuracy", which at K=8 is a very different and
    much larger quantity.
    """

    correct_rollouts: int = 0
    generated_rollouts: int = 0
    problems_any_correct: int = 0
    total_problems: int = 0
    retained_rollouts: int = 0
    #: Resource axes. A learning curve must be plotted against these, not just
    #: against rounds: arms that select differently consume different budgets.
    optimizer_steps: int = 0
    response_tokens_trained: int = 0
    generated_tokens: int = 0
    teacher_forward_tokens: int = 0
    elapsed_seconds: float = 0.0
    peak_memory_bytes: int | None = None
    verifier_outcomes: dict[str, int] = field(default_factory=dict)

    @property
    def pass_at_1(self) -> float:
        """Expected accuracy of a single sampled response."""

        if not self.generated_rollouts:
            return 0.0
        return self.correct_rollouts / self.generated_rollouts

    @property
    def pass_at_k(self) -> float:
        """Fraction of problems solved by at least one of K rollouts."""

        if not self.total_problems:
            return 0.0
        return self.problems_any_correct / self.total_problems

    @property
    def accuracy(self) -> float:
        """Alias for :attr:`pass_at_1`, the reported headline metric."""

        return self.pass_at_1

    @property
    def pass_at_1_interval(self) -> tuple[float, float]:
        return wilson_interval(self.correct_rollouts, self.generated_rollouts)

    @property
    def retained_rollout_rate(self) -> float:
        if not self.generated_rollouts:
            return 0.0
        return self.retained_rollouts / self.generated_rollouts

    @property
    def throughput_rollouts_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.generated_rollouts / self.elapsed_seconds

    def as_dict(self) -> dict[str, object]:
        low, high = self.pass_at_1_interval
        return {
            "pass_at_1": self.pass_at_1,
            "pass_at_1_ci_low": low,
            "pass_at_1_ci_high": high,
            "pass_at_k": self.pass_at_k,
            "correct_rollouts": self.correct_rollouts,
            "generated_rollouts": self.generated_rollouts,
            "problems_any_correct": self.problems_any_correct,
            "total_problems": self.total_problems,
            "retained_rollouts": self.retained_rollouts,
            "retained_rollout_rate": self.retained_rollout_rate,
            "optimizer_steps": self.optimizer_steps,
            "response_tokens_trained": self.response_tokens_trained,
            "generated_tokens": self.generated_tokens,
            "teacher_forward_tokens": self.teacher_forward_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "throughput_rollouts_per_second": self.throughput_rollouts_per_second,
            "peak_memory_bytes": self.peak_memory_bytes,
            "verifier_outcomes": dict(self.verifier_outcomes),
        }


__all__ = ["EfficiencyMetrics", "mcnemar_pvalue", "wilson_interval"]
