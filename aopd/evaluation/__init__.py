"""Held-out evaluation and efficiency metrics."""

from .evaluator import (
    EvaluationConfig,
    EvaluationResult,
    evaluate_dataset,
    evaluate_rollout_groups,
    paired_comparison,
)
from .metrics import EfficiencyMetrics, mcnemar_pvalue, wilson_interval

__all__ = [
    "EfficiencyMetrics",
    "EvaluationConfig",
    "EvaluationResult",
    "evaluate_dataset",
    "evaluate_rollout_groups",
    "mcnemar_pvalue",
    "paired_comparison",
    "wilson_interval",
]
