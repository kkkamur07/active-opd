"""Explicit lazy OpenThoughts dataset wrapper."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from .datasets import MathExample, example_from_record, load_openthoughts


@dataclass(frozen=True)
class OpenThoughtsConfig:
    dataset_name: str = "open-thoughts/OpenThoughts-114k"
    split: str = "train"
    streaming: bool = True
    cache_dir: str | None = None
    limit: int | None = None


class OpenThoughtsDataset:
    """A re-iterable, non-materializing dataset adapter."""

    def __init__(
        self,
        config: OpenThoughtsConfig | Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(config, OpenThoughtsConfig):
            self.config = config
        else:
            values = config or {}
            names = set(OpenThoughtsConfig.__dataclass_fields__)
            self.config = OpenThoughtsConfig(
                **{name: values[name] for name in names if name in values}
            )

    def records(self) -> Iterator[Mapping[str, Any]]:
        """Load records only when iteration begins."""

        records = load_openthoughts(
            dataset_name=self.config.dataset_name,
            split=self.config.split,
            streaming=self.config.streaming,
            cache_dir=self.config.cache_dir,
        )
        for index, record in enumerate(records):
            if self.config.limit is not None and index >= self.config.limit:
                break
            yield record

    def examples(self) -> Iterator[MathExample]:
        """Yield canonical examples lazily."""

        for record in self.records():
            yield example_from_record(record)

    def __iter__(self) -> Iterator[MathExample]:
        return self.examples()


__all__ = ["OpenThoughtsConfig", "OpenThoughtsDataset"]
