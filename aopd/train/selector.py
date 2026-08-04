"""Rollout selection policies and auditable verification accounting.

Every selector returns per-rollout weights alongside the selected records.  A
hard filter is the special case ``weight in {0, 1}``; a future acquisition
score is the same interface with continuous weights, which is what keeps the
filter-vs-score comparison controlled rather than a rewrite.

Only rollouts whose answer the verifier actually decided (``correct`` /
``wrong``) can ever be retained.  ``malformed``, ``truncated``, ``unverified``
and ``skipped`` rollouts have no trustworthy label and are counted, reported,
and dropped.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from aopd.data.answers import VerificationOutcome
from aopd.data.rollouts import Rollout, VerificationSummary, summarize_verification

SelectionPolicy = Literal["verified_wrong", "all", "random", "correct"]


@dataclass(frozen=True)
class FilteringConfig:
    """Filtering controls mirrored by ``configs/filtering``."""

    policy: SelectionPolicy | str = "verified_wrong"
    #: Selection budget in rollouts. Required for a fair random baseline: the
    #: control must match the treatment's *count*, not just its policy.
    budget: int | None = None
    seed: int = 42

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> FilteringConfig:
        names = {"policy", "budget", "seed"}
        unknown = set(config) - names
        if unknown:
            raise ValueError(
                f"Unknown filtering option(s): {sorted(unknown)}. Known: {sorted(names)}."
            )
        return cls(**{name: config[name] for name in names if name in config})


@dataclass(frozen=True)
class SelectionResult:
    """Selected records, their weights, and counts for every gate outcome."""

    selected: tuple[Rollout, ...]
    summary: VerificationSummary
    policy: str
    weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.weights:
            object.__setattr__(self, "weights", (1.0,) * len(self.selected))
        elif len(self.weights) != len(self.selected):
            raise ValueError("weights must align one-to-one with selected rollouts.")

    @property
    def retained_rollout_rate(self) -> float:
        return self.summary.retained_rate

    @property
    def response_tokens(self) -> int:
        """Total generated tokens in the selection -- the real training budget."""

        return sum(int(rollout.response_length or 0) for rollout in self.selected)


class _BaseSelector:
    policy = "base"
    retain_outcomes: tuple[VerificationOutcome, ...] = ()

    def select(self, rollouts: Iterable[Rollout]) -> SelectionResult:
        selected, summary = summarize_verification(
            list(rollouts), retain_outcomes=self.retain_outcomes
        )
        return SelectionResult(tuple(selected), summary, self.policy)


class VerifiedWrongSelector(_BaseSelector):
    """Active OPD gate: keep rollouts the verifier decided are wrong."""

    policy = "verified_wrong"
    retain_outcomes = ("wrong",)

    def __init__(self, config: FilteringConfig | Mapping[str, Any] | None = None) -> None:
        self.config = (
            config
            if isinstance(config, FilteringConfig)
            else FilteringConfig.from_mapping(config or {})
        )


class AllRolloutSelector(_BaseSelector):
    """Standard OPD baseline: every rollout with a decided verdict."""

    policy = "all"
    retain_outcomes = ("correct", "wrong")


class CorrectRolloutSelector(_BaseSelector):
    """Correctness-only baseline retained for controlled comparisons."""

    policy = "correct"
    retain_outcomes = ("correct",)


class RandomRolloutSelector:
    """Budget-matched random control.

    This is the baseline that separates "*which* rollouts were chosen matters"
    from "*fewer, fresher* updates matter", so the budget is mandatory and the
    RNG is owned by the instance -- rebuilding it inside ``select`` would make
    every round pick the same positions.
    """

    policy = "random"
    retain_outcomes: tuple[VerificationOutcome, ...] = ("correct", "wrong")

    def __init__(self, budget: int | None = None, seed: int = 42) -> None:
        if budget is not None and budget < 0:
            raise ValueError("budget must be non-negative or None.")
        self.budget = budget
        self.seed = seed
        self._rng = random.Random(seed)

    def select(self, rollouts: Iterable[Rollout]) -> SelectionResult:
        selected, summary = summarize_verification(
            list(rollouts), retain_outcomes=self.retain_outcomes
        )
        if self.budget is not None and len(selected) > self.budget:
            selected = self._rng.sample(selected, self.budget)
        return SelectionResult(tuple(selected), summary, self.policy)


def match_budget(
    result: SelectionResult,
    budget: int,
    *,
    seed: int = 0,
) -> SelectionResult:
    """Down- or up-sample a selection to exactly ``budget`` rollouts.

    Arms must consume the same number of optimizer steps for a learning-rate
    curve to be comparable. When a pool is smaller than the budget it is
    sampled with replacement rather than silently running short.
    """

    if budget < 0:
        raise ValueError("budget must be non-negative.")
    records = list(result.selected)
    if not records or len(records) == budget:
        return result
    rng = random.Random(seed)
    if len(records) > budget:
        matched = rng.sample(records, budget)
    else:
        matched = records + [rng.choice(records) for _ in range(budget - len(records))]
    return SelectionResult(tuple(matched), result.summary, result.policy)


def make_selector(
    policy: str,
    *,
    budget: int | None = None,
    seed: int = 42,
    config: Mapping[str, Any] | None = None,
) -> Any:
    """Construct a selector by Hydra/YAML policy name."""

    normalized = policy.lower().replace("-", "_")
    if normalized == "verified_wrong":
        return VerifiedWrongSelector(config)
    if normalized == "all":
        return AllRolloutSelector()
    if normalized == "correct":
        return CorrectRolloutSelector()
    if normalized == "random":
        return RandomRolloutSelector(budget=budget, seed=seed)
    raise ValueError(
        f"Unknown rollout selection policy: {policy!r}. "
        "Known policies: all, correct, random, verified_wrong."
    )


def select_verified_wrong(rollouts: Iterable[Rollout]) -> SelectionResult:
    """Convenience API used by small experiments and tests."""

    return VerifiedWrongSelector().select(rollouts)


__all__ = [
    "AllRolloutSelector",
    "CorrectRolloutSelector",
    "FilteringConfig",
    "RandomRolloutSelector",
    "SelectionResult",
    "VerifiedWrongSelector",
    "make_selector",
    "match_budget",
    "select_verified_wrong",
]
