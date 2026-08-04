import pytest

from aopd.data.rollouts import Rollout
from aopd.evaluation import (
    evaluate_rollout_groups,
    mcnemar_pvalue,
    paired_comparison,
    wilson_interval,
)


def _group(problem_id, responses, reference="1"):
    return [
        Rollout(
            "p",
            response,
            reference,
            rollout_id=f"{problem_id}:{index}",
            metadata={"problem_id": problem_id},
        )
        for index, response in enumerate(responses)
    ]


def test_pass_at_1_and_pass_at_k_are_reported_separately():
    """pass@k was previously reported as 'accuracy'; at K=8 that is a very
    different and much larger number."""

    groups = [
        _group("a", [r"</think>\boxed{1}", r"</think>\boxed{2}"]),  # 1 of 2
        _group("b", [r"</think>\boxed{2}", r"</think>\boxed{2}"]),  # 0 of 2
    ]

    result = evaluate_rollout_groups(groups)

    assert result.metrics.pass_at_1 == pytest.approx(0.25)
    assert result.metrics.pass_at_k == pytest.approx(0.5)
    assert result.metrics.accuracy == result.metrics.pass_at_1


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(35, 100)
    assert low < 0.35 < high
    # A wider interval at smaller n is exactly why n=128 is underpowered here.
    narrow = wilson_interval(350, 1000)
    assert (high - low) > (narrow[1] - narrow[0])


def test_wilson_interval_handles_degenerate_inputs():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert high > 0


def test_mcnemar_only_reacts_to_discordant_pairs():
    assert mcnemar_pvalue(0, 0) == 1.0
    assert mcnemar_pvalue(10, 10) == pytest.approx(1.0)
    assert mcnemar_pvalue(10, 0) < 0.01


def test_paired_comparison_uses_shared_problems():
    first = evaluate_rollout_groups(
        [_group("a", [r"</think>\boxed{1}"]), _group("b", [r"</think>\boxed{1}"])]
    )
    second = evaluate_rollout_groups(
        [_group("a", [r"</think>\boxed{9}"]), _group("b", [r"</think>\boxed{1}"])]
    )

    comparison = paired_comparison(first, second)

    assert comparison["shared_problems"] == 2
    assert comparison["only_first"] == 1
    assert comparison["only_second"] == 0
    assert comparison["accuracy_delta"] > 0


def test_truncated_evaluation_rollouts_count_as_incorrect_not_dropped():
    groups = [_group("a", ["<think>never finished"])]
    for rollout in groups[0]:
        rollout.truncated = True

    result = evaluate_rollout_groups(groups)

    assert result.metrics.pass_at_1 == 0.0
    assert result.metrics.generated_rollouts == 1
    assert result.metrics.verifier_outcomes["truncated"] == 1
