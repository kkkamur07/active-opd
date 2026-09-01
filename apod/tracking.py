"""Weights & Biases tracking: one W&B run per arm, shared by every process
that logs for that arm.

The driver (refresh evals) and the train stage (per-step scalars) are
separate processes, launched one after the other, that must land in the SAME
W&B run: the run ``id`` is derived deterministically from the run dir and the
arm, and every ``init`` resumes it (``resume="allow"``). All scalars are
logged against ``global_step`` -- the training step counted across refreshes
-- declared as the x-axis with ``define_metric``, so processes never fight
over W&B's own monotonic step counter.

Settings come from ``conf/tracking.yaml`` (``cfg.tracking``):

    mode: offline | online | disabled   (default offline: never blocks on network)
    project, entity, experiment (tag; null -> run dir name), dir (null -> <run_dir>/wandb)

Offline runs are uploaded later with ``wandb sync <run_dir>/wandb/offline-run-*``
(every process writes its own offline-run dir; they all carry the same run id,
so syncing them fills in one run). Every function is a no-op when wandb is
not installed, ``mode`` is ``disabled``, or the process is a non-zero
torchrun rank, so the pipeline never depends on the package being present.

API (see docs/pipeline.md):

    init(cfg, run_dir, arm) -> bool      start/resume the arm's run in this process
    log_step(step, metrics)              per-training-step scalars (keys get "train/")
    log_refresh(step, metrics)           refresh eval scalars, keys as given ("eval/...")
    finish()                             flush and close this process's handle
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from loguru import logger

_run: Any = None
_MODES = ("online", "offline", "disabled")


def run_id(run_dir: Path | str, arm: str) -> str:
    """Deterministic W&B run id for (run dir name, arm): a readable slug plus a
    short hash so truncation to W&B's id length can never collide."""
    key = f"{Path(run_dir).name}/{arm}"
    slug = re.sub(r"[^a-z0-9_-]+", "-", key.lower()).strip("-")[:48]
    return f"{slug}-{hashlib.sha1(key.encode()).hexdigest()[:8]}"


def _to_plain(cfg) -> dict:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return dict(cfg)


def init(cfg, run_dir: Path | str, arm: str) -> bool:
    """Start (or resume) the W&B run for ``arm`` of ``run_dir`` in this process.

    Returns True when a run is live. False -- with the reason logged -- when
    tracking is disabled, wandb is missing, or this is a non-zero rank.
    """
    global _run
    if _run is not None:
        return True
    if int(os.environ.get("RANK", "0")) != 0:
        return False
    settings = _to_plain(cfg).get("tracking") or {}
    mode = str(settings.get("mode") or "offline")
    if mode not in _MODES:
        raise ValueError(f"tracking.mode must be one of {_MODES}; got {mode!r}")
    if mode == "disabled":
        return False
    try:
        import wandb
    except ImportError:
        logger.warning("tracking.mode={} but wandb is not installed; tracking disabled", mode)
        return False

    run_dir = Path(run_dir)
    # wandb creates its own ``wandb/`` under ``dir``: runs land in
    # <run_dir>/wandb/offline-run-*-<id> (or run-* when online).
    root = Path(settings.get("dir") or run_dir)
    root.mkdir(parents=True, exist_ok=True)
    plain_cfg = _to_plain(cfg)
    cap = (plain_cfg.get("sampling") or {}).get("max_new_tokens")
    experiment = settings.get("experiment") or run_dir.name
    tags = [f"cap{cap}" if cap is not None else "cap-unknown", str(experiment)]
    _run = wandb.init(
        project=str(settings.get("project") or "apod"),
        entity=settings.get("entity") or None,
        group=run_dir.name,
        name=f"{run_dir.name}/{arm}",
        job_type=arm,
        tags=tags,
        config=plain_cfg,
        id=run_id(run_dir, arm),
        resume="allow",
        mode=mode,
        dir=str(root),
    )
    # One x-axis for everything: the global training step (across refreshes),
    # passed explicitly with each log call. W&B's internal step is left alone,
    # so a later process resuming the run can never trip its monotonic check.
    _run.define_metric("global_step")
    _run.define_metric("train/*", step_metric="global_step")
    _run.define_metric("eval/*", step_metric="global_step")
    logger.info("wandb {} run {} ({}) -> {}", mode, _run.name, _run.id, _run.dir)
    return True


def _log(step: int, metrics: dict[str, Any]) -> None:
    if _run is None:
        return
    _run.log({**metrics, "global_step": int(step)})


def log_step(step: int, metrics: dict[str, Any]) -> None:
    """Per-training-step scalars at global ``step``. Keys without a namespace
    are logged under ``train/`` (``loss`` -> ``train/loss``)."""
    _log(step, {k if "/" in k else f"train/{k}": v for k, v in metrics.items()})


def log_refresh(step: int, metrics: dict[str, Any]) -> None:
    """Refresh-time eval scalars at global ``step``, keys as given, e.g.
    ``eval/math500/strict_avg4``, ``eval/aime/pass16``."""
    _log(step, dict(metrics))


def finish() -> None:
    """Flush and close this process's handle on the run (the run itself stays
    resumable by the next process)."""
    global _run
    if _run is None:
        return
    _run.finish()
    _run = None
