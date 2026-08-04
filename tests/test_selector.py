import pytest

from aopd.data.rollouts import Rollout, summarize_verification
from aopd.train.selector import (
    AllRolloutSelector,
    CorrectRolloutSelector,
    RandomRolloutSelector,
    VerifiedWrongSelector,
    match_budget,
)


def _rollouts():
    return [
        Rollout("p", r"</think>\boxed{1}", "1"),  # correct
        Rollout("p", r"</think>\boxed{2}", "1"),  # wrong
        Rollout("p", "</think>No final answer", "1"),  # malformed
        Rollout("p", r"</think>\boxed{3}", None),  # skipped: no reference
        Rollout("p", "<think>still reasoning", "1", truncated=True),  # truncated
    ]


def test_verified_wrong_selector_retains_only_decided_wrong_answers():
    result = VerifiedWrongSelector().select(_rollouts())

    assert [item.response for item in result.selected] == [r"</think>\boxed{2}"]
    assert result.summary.correct == 1
    assert result.summary.wrong == 1
    assert result.summary.malformed == 1
    assert result.summary.skipped == 1
    assert result.summary.truncated == 1


def test_truncated_rollouts_are_never_training_signal():
    """A trace that ran out of budget has no verdict, only a length."""

    for selector in (VerifiedWrongSelector(), AllRolloutSelector(), CorrectRolloutSelector()):
        result = selector.select(_rollouts())
        assert all(not item.truncated for item in result.selected), selector.policy


def test_retaining_an_undecided_outcome_is_rejected():
    with pytest.raises(ValueError, match="must not include"):
        summarize_verification(_rollouts(), retain_outcomes=("wrong", "truncated"))


def test_standard_baseline_retains_every_decided_rollout():
    result = AllRolloutSelector().select(_rollouts())

    assert len(result.selected) == 2  # the correct one and the wrong one
    assert result.summary.usable_rate == pytest.approx(2 / 5)


def test_random_selector_varies_across_rounds_but_is_seed_reproducible():
    """The RNG used to be rebuilt inside select(), so every round returned the
    same positions and the 'random' control was not random over time."""

    selector = RandomRolloutSelector(budget=1, seed=7)
    draws = [selector.select(_rollouts()).selected[0].response for _ in range(6)]
    assert len(set(draws)) > 1

    repeat = RandomRolloutSelector(budget=1, seed=7)
    assert [
        repeat.select(_rollouts()).selected[0].response for _ in range(6)
    ] == draws


def test_random_selector_without_a_budget_is_not_silently_the_all_baseline():
    unbudgeted = RandomRolloutSelector().select(_rollouts())
    everything = AllRolloutSelector().select(_rollouts())

    # Documented equivalence: without a budget there is nothing to sample down
    # to. The ablation script must therefore always pass one.
    assert len(unbudgeted.selected) == len(everything.selected)


def test_match_budget_makes_arms_consume_the_same_number_of_steps():
    result = AllRolloutSelector().select(_rollouts())

    smaller = match_budget(result, 1, seed=0)
    larger = match_budget(result, 5, seed=0)

    assert len(smaller.selected) == 1
    assert len(larger.selected) == 5  # sampled with replacement, not truncated


def test_selection_reports_the_token_budget_not_just_the_rollout_count():
    rollouts = _rollouts()
    for rollout in rollouts:
        rollout.response_length = 10

    result = AllRolloutSelector().select(rollouts)

    assert result.response_tokens == 20
