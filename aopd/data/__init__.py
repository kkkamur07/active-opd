"""Lazy datasets, answer verification, and rollout records."""

from .answers import (
    AnswerExtraction,
    VerificationOutcome,
    VerificationResult,
    extract_answer,
    extract_final_answer,
    normalize_answer,
    verify_exact_answer,
)
from .datasets import (
    MathExample,
    example_from_record,
    iter_examples,
    load_math500,
    load_openthoughts,
)
from .openthoughts import OpenThoughtsConfig, OpenThoughtsDataset
from .rollouts import Rollout, VerificationSummary, summarize_verification

__all__ = [
    "AnswerExtraction",
    "MathExample",
    "OpenThoughtsConfig",
    "OpenThoughtsDataset",
    "Rollout",
    "VerificationOutcome",
    "VerificationResult",
    "VerificationSummary",
    "example_from_record",
    "extract_answer",
    "extract_final_answer",
    "iter_examples",
    "load_math500",
    "load_openthoughts",
    "normalize_answer",
    "summarize_verification",
    "verify_exact_answer",
]
