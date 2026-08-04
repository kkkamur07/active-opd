"""Small JSONL metrics logger with no experiment-tracking dependency."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (AttributeError, TypeError, ValueError):
            pass
    return str(value)


class JsonlMetricsLogger:
    """Append one JSON object per training/evaluation event."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **metrics: Any) -> None:
        record = {
            "event": event,
            "timestamp": datetime.now(UTC).isoformat(),
            **{key: _json_safe(value) for key, value in metrics.items()},
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = ["JsonlMetricsLogger"]
