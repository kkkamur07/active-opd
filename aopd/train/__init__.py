"""Rollout collection, selection, and optimization orchestration."""

from .batching import batch_builder, build_token_batch, iter_micro_batches
from .rollouts import RolloutCollectionConfig, RolloutCollector, collect_rollouts
from .selector import (
    AllRolloutSelector,
    CorrectRolloutSelector,
    FilteringConfig,
    RandomRolloutSelector,
    SelectionResult,
    VerifiedWrongSelector,
    make_selector,
    match_budget,
    select_verified_wrong,
)
from .trainer import OPDTrainer, TrainerConfig, TrainerState

__all__ = [
    "AllRolloutSelector",
    "CorrectRolloutSelector",
    "FilteringConfig",
    "OPDTrainer",
    "RandomRolloutSelector",
    "RolloutCollectionConfig",
    "RolloutCollector",
    "SelectionResult",
    "TrainerConfig",
    "TrainerState",
    "VerifiedWrongSelector",
    "batch_builder",
    "build_token_batch",
    "collect_rollouts",
    "iter_micro_batches",
    "make_selector",
    "match_budget",
    "select_verified_wrong",
]
