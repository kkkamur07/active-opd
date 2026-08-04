"""Tests for the budget-matching that makes the ablation interpretable."""

from scripts.experiments.run_filter_ablation import _plan, _parse_args, validate_fairness


def _state(steps, tokens, generated=1000):
    return {
        "state": {
            "optimizer_steps": steps,
            "response_tokens_trained": tokens,
            "generated_rollouts": generated,
        }
    }


def test_matched_budgets_pass_the_fairness_checks():
    results = {
        "all": _state(320, 100_000),
        "random": _state(320, 103_000),
        "verified_wrong": _state(320, 98_000),
    }

    report = validate_fairness(results)

    assert report["checks"]["all_required_checks_pass"] is True


def test_unequal_optimizer_steps_fail():
    """The committed pilot: standard took 8 steps, active took 2, and the old
    fairness block still reported PASS."""

    results = {
        "all": _state(8, 50_000),
        "verified_wrong": _state(2, 12_000),
    }

    report = validate_fairness(results)

    assert report["checks"]["same_optimizer_steps"] is False
    assert report["checks"]["all_required_checks_pass"] is False


def test_matched_steps_but_lopsided_token_budgets_fail():
    """Equal step counts are not enough: an arm that selects longer rollouts
    backpropagates more text per step."""

    results = {
        "all": _state(320, 100_000),
        "verified_wrong": _state(320, 160_000),
    }

    report = validate_fairness(results)

    assert report["checks"]["same_optimizer_steps"] is True
    assert report["checks"]["response_tokens_within_10pct"] is False
    assert report["checks"]["all_required_checks_pass"] is False


def test_unequal_generation_budgets_fail():
    """Selection must not change how much was sampled: all arms generate the
    same rollouts, so selection saves training compute, not sampling compute."""

    results = {
        "all": _state(320, 100_000, generated=2000),
        "verified_wrong": _state(320, 100_000, generated=1000),
    }

    assert validate_fairness(results)["checks"]["same_rollouts_generated"] is False


def test_plan_reports_the_budget_each_arm_will_consume():
    options = _parse_args(
        ["--train-prompts", "64", "--k", "4", "--rounds", "20", "--select-budget", "16"]
    )

    plan = _plan(options)

    assert plan["rollouts_generated_per_round_per_arm"] == 256
    assert plan["rollouts_trained_per_round_per_arm"] == 16
    assert plan["optimizer_steps_per_arm"] == 320


def test_every_arm_is_matched_to_the_same_training_budget():
    from aopd.data.rollouts import Rollout
    from aopd.train import make_selector, match_budget

    rollouts = [
        Rollout("p", rf"</think>\boxed{{{value}}}", "1", rollout_id=str(index))
        for index, value in enumerate([1, 1, 2, 3, 4, 1, 5, 6])
    ]

    sizes = []
    for arm in ("all", "random", "verified_wrong"):
        selector = make_selector(arm, budget=4, seed=0)
        matched = match_budget(selector.select(rollouts), 4, seed=0)
        sizes.append(len(matched.selected))

    assert sizes == [4, 4, 4]
