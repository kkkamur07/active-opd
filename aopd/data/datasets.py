"""Lazy dataset adapters used by rollout collection and evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from .answers import extract_final_answer


@dataclass(frozen=True)
class MathExample:
    """Canonical prompt/reference pair independent of a dataset backend."""

    prompt: str
    reference_answer: str | None = None
    problem_id: str | None = None
    metadata: Mapping[str, Any] | None = None


def _first_value(row: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def _conversation_value(
    row: Mapping[str, Any],
    *,
    roles: tuple[str, ...],
    value_fields: tuple[str, ...],
) -> str | None:
    conversations = row.get("conversations", row.get("messages"))
    if not isinstance(conversations, (list, tuple)):
        return None
    for message in conversations:
        if not isinstance(message, Mapping):
            continue
        role = message.get("from", message.get("role"))
        if str(role).lower() not in roles:
            continue
        value = _first_value(message, value_fields)
        if value is not None:
            return str(value)
    return None


#: Columns that hold a genuine ground-truth answer, in order of preference.
ANSWER_FIELDS: tuple[str, ...] = (
    "answer",
    "final_answer",
    "target",
    "ground_truth",
    "gt_answer",
    "expected_answer",
)

#: Columns that hold a worked solution. An answer scraped out of one of these
#: is a parse of someone else's reasoning, not ground truth.
SOLUTION_FIELDS: tuple[str, ...] = ("solution", "reasoning", "completion")


def example_from_record(
    row: Mapping[str, Any],
    *,
    prompt_fields: tuple[str, ...] = ("problem", "question", "prompt", "input"),
    answer_fields: tuple[str, ...] = ANSWER_FIELDS,
    allow_solution_fallback: bool = False,
    keep_metadata: bool = False,
) -> MathExample:
    """Convert a MATH-style record to ``MathExample``.

    By default the reference answer must come from a real answer column.  The
    previous behaviour -- falling back to running the answer *extractor* over a
    reasoning trace -- silently produced references such as ``"'yes' if"`` on
    datasets that have no answer column at all, and every downstream
    correctness label inherited that noise.  Set ``allow_solution_fallback``
    only when the solution field is known to end in ``\\boxed{...}``, and note
    that the extraction is still only as good as the trace.
    """

    prompt = _first_value(row, prompt_fields)
    if prompt is None:
        prompt = _conversation_value(
            row,
            roles=("user", "human"),
            value_fields=("value", "content", "text"),
        )
    if prompt is None:
        raise ValueError(
            f"Dataset record has no prompt field; tried {prompt_fields!r}. "
            f"Available columns: {sorted(row)}."
        )

    reference = _first_value(row, answer_fields)
    if reference is None and allow_solution_fallback:
        solution = _first_value(row, SOLUTION_FIELDS)
        if solution is None:
            solution = _conversation_value(
                row,
                roles=("assistant", "gpt", "bot"),
                value_fields=("value", "content", "text"),
            )
        if solution is not None:
            extracted = extract_final_answer(
                str(solution), require_closed_reasoning=False
            )
            # Only a boxed answer is trustworthy enough to use as ground truth.
            if extracted.status == "ok" and extracted.source == "boxed":
                reference = extracted.answer
    if reference is None:
        raise ValueError(
            "Dataset record has no ground-truth answer column; tried "
            f"{answer_fields!r}. Available columns: {sorted(row)}. Use a dataset "
            "with a real answer column rather than scraping reasoning traces."
        )

    problem_id = _first_value(row, ("id", "problem_id", "unique_id", "index"))
    return MathExample(
        prompt=str(prompt),
        reference_answer=str(reference),
        problem_id=str(problem_id) if problem_id is not None else None,
        metadata=dict(row) if keep_metadata else None,
    )


def iter_examples(
    records: Iterable[Mapping[str, Any]],
    *,
    limit: int | None = None,
    skip_invalid: bool = False,
    **example_kwargs: Any,
) -> Iterator[MathExample]:
    """Adapt an iterable without materializing the full dataset.

    ``limit`` counts *yielded* examples, not scanned records, so skipping
    invalid rows cannot silently shrink the requested sample.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None.")
    yielded = 0
    for row in records:
        if limit is not None and yielded >= limit:
            return
        try:
            example = example_from_record(row, **example_kwargs)
        except ValueError:
            if skip_invalid:
                continue
            raise
        yielded += 1
        yield example


#: Math training corpora that ship a real ground-truth answer column.
#: OpenThoughts-114k is deliberately *not* here: its default split has only
#: ``system``/``conversations``, is sorted by domain (the first ~21k records are
#: competitive programming), and yields no usable reference answer.
MATH_TRAIN_DATASETS: dict[str, dict[str, Any]] = {
    "openr1": {
        "dataset_name": "open-r1/OpenR1-Math-220k",
        "config_name": "default",
        "split": "train",
        "answer_fields": ("answer",),
        "prompt_fields": ("problem",),
    },
    "math": {
        "dataset_name": "EleutherAI/hendrycks_math",
        "config_name": "algebra",
        "split": "train",
        "answer_fields": ANSWER_FIELDS,
        "prompt_fields": ("problem",),
        "allow_solution_fallback": True,
    },
}


def load_math_training_set(
    preset: str = "openr1",
    *,
    streaming: bool = True,
    cache_dir: str | None = None,
    shuffle_buffer: int | None = 10_000,
    seed: int = 42,
    **overrides: Any,
) -> tuple[Iterable[Mapping[str, Any]], dict[str, Any]]:
    """Load a math-only training corpus plus the fields needed to parse it.

    Returns ``(records, example_kwargs)``. Shuffling is on by default because
    these corpora are stored grouped by source/difficulty, so reading from
    record 0 samples one narrow slice.
    """

    try:
        settings = dict(MATH_TRAIN_DATASETS[preset])
    except KeyError:
        raise ValueError(
            f"Unknown math dataset preset {preset!r}. "
            f"Known presets: {sorted(MATH_TRAIN_DATASETS)}."
        ) from None
    settings.update(overrides)
    example_kwargs = {
        key: settings.pop(key)
        for key in ("answer_fields", "prompt_fields", "allow_solution_fallback")
        if key in settings
    }

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Math dataset loading requires the optional 'datasets' dependency."
        ) from exc

    kwargs: dict[str, Any] = {
        "split": settings.pop("split"),
        "streaming": streaming,
    }
    config_name = settings.pop("config_name", None)
    if config_name is not None:
        kwargs["name"] = config_name
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    kwargs.update(settings)
    records = load_dataset(kwargs.pop("dataset_name"), **kwargs)
    if shuffle_buffer:
        try:
            records = records.shuffle(seed=seed, buffer_size=shuffle_buffer)
        except TypeError:  # pragma: no cover - non-streaming datasets
            records = records.shuffle(seed=seed)
    return records, example_kwargs


def load_dataset_records(
    *,
    dataset_name: str,
    config_name: str | None = None,
    split: str = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
    **load_kwargs: Any,
) -> Iterable[Mapping[str, Any]]:
    """Return a lazy iterator of raw rows from any Hugging Face dataset.

    Used by the inspection scripts, which need the unparsed row (including
    rows with no usable answer column) rather than a ``MathExample``.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Dataset loading requires the optional 'datasets' dependency."
        ) from exc

    kwargs: dict[str, Any] = {"split": split, "streaming": streaming, **load_kwargs}
    if config_name is not None:
        kwargs["name"] = config_name
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    return load_dataset(dataset_name, **kwargs)


def load_openthoughts(
    *,
    dataset_name: str = "open-thoughts/OpenThoughts-114k",
    config_name: str | None = None,
    split: str = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
    **load_kwargs: Any,
) -> Iterable[Mapping[str, Any]]:
    """Return a lazy OpenThoughts dataset iterator.

    ``datasets`` is optional so importing the package and running unit tests
    never downloads data.  The first call to this function is the explicit
    data-loading boundary.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "OpenThoughts loading requires the optional 'datasets' dependency."
        ) from exc

    kwargs: dict[str, Any] = {
        "split": split,
        "streaming": streaming,
        **load_kwargs,
    }
    if config_name is not None:
        kwargs["name"] = config_name
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    return load_dataset(dataset_name, **kwargs)


def load_math500(
    *,
    dataset_name: str = "HuggingFaceH4/MATH-500",
    split: str = "test",
    streaming: bool = True,
    cache_dir: str | None = None,
    **load_kwargs: Any,
) -> Iterable[Mapping[str, Any]]:
    """Return a lazy MATH-500-style dataset iterator."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "MATH-500 loading requires the optional 'datasets' dependency."
        ) from exc
    kwargs: dict[str, Any] = {
        "split": split,
        "streaming": streaming,
        **load_kwargs,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    return load_dataset(dataset_name, **kwargs)


__all__ = [
    "ANSWER_FIELDS",
    "MATH_TRAIN_DATASETS",
    "MathExample",
    "example_from_record",
    "iter_examples",
    "load_math500",
    "load_dataset_records",
    "load_math_training_set",
    "load_openthoughts",
]
