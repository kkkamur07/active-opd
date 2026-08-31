"""GKD training stage: one reverse-KL pass over a round's selected trajectories.

The driver launches this under ``torchrun --nproc_per_node=<num_gpus>`` with
both GPUs visible, so training is data-parallel (DDP): student and frozen
teacher are replicated per rank, each rank trains on its sampler shard, and
the only cross-GPU traffic is the student gradient all-reduce (~4 GB bf16 per
step, overlapped with backward; the frozen teacher never syncs).
``per_device_train_batch_size`` is per rank -- the config halves
``gradient_accumulation_steps`` relative to the single-GPU setting so the
effective batch stays the same. With one GPU it degrades to the plain
single-process run. Loads the round's starting student, distills from
the frozen teacher (``cfg.model.teacher_id`` — Qwen/Qwen3.5-9B) with TRL's
GKDTrainer at beta=1.0 / lmbda=0.0 (pure reverse KL on our provided
trajectories, no in-trainer generation), and writes ``checkpoint/`` for the
next round. See docs/pipeline.md for the contract and conf/train/gkd.yaml for
the verified-against-source hyperparameters.

    python -m apod.stages.train --run-dir D --arm A --round R
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

from apod.datasets.io import read_jsonl, write_jsonl
from apod.paths import round_dir
from apod.stages.common import parse_stage_args, stage_parser


def parse_args() -> argparse.Namespace:
    return parse_stage_args(stage_parser(description=__doc__, needs_shards=False))


def build_rows(
    rdir: Path,
    selected: list[dict],
    *,
    eos_ids: set[int],
    append_eos_id: int,
    max_length: int,
) -> tuple[list[dict], int, int]:
    """Turn selected.jsonl + per-example npz token dumps into GKDTrainer rows.

    Each row: ``input_ids`` (prompt+response, the exact ids vLLM produced),
    ``completion_mask`` (0 over the prompt, 1 over the response), and
    ``prompt`` (text — the collator requires the key even pre-tokenized).
    Returns (rows, completion_tokens, tail_truncated_rows).
    """

    by_example: dict[int, list[dict]] = defaultdict(list)
    for row in selected:
        by_example[row["example_index"]].append(row)

    rows: list[dict] = []
    completion_tokens = 0
    tail_truncated = 0
    for example_index in sorted(by_example):
        npz_path = rdir / "rollouts" / "tokens" / f"example_{example_index:05d}.npz"
        with np.load(npz_path, allow_pickle=True) as data:
            input_ids = data["input_ids"]
            prompt_length = int(data["prompt_length"])
            response_lengths = data["response_lengths"]
            truncated = data["truncated"]
        for sel in by_example[example_index]:
            j = sel["rollout_index"]
            length = prompt_length + int(response_lengths[j])
            ids = [int(t) for t in input_ids[j, :length]]
            # The loss needs the stop token: append EOS if generation finished
            # normally but the stored ids do not already end with one. The
            # appended id is the tokenizer's canonical EOS explicitly, not an
            # arbitrary member of the set.
            if not bool(truncated[j]) and ids[-1] not in eos_ids:
                ids.append(append_eos_id)
            if len(ids) > max_length:
                # Tail truncation drops the highest-signal tokens (ending +
                # EOS); counted into summary.json so it can't stay invisible.
                logger.warning(
                    "example {} rollout {}: {} tokens > train.max_length {}, truncating tail",
                    example_index, j, len(ids), max_length,
                )
                ids = ids[:max_length]
                tail_truncated += 1
            mask = [0] * prompt_length + [1] * (len(ids) - prompt_length)
            rows.append({"input_ids": ids, "completion_mask": mask, "prompt": ""})
            completion_tokens += len(ids) - prompt_length
    return rows, completion_tokens, tail_truncated


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    rdir = round_dir(run_dir, args.arm, args.round_index)
    train_dir = rdir / "train"
    checkpoint_dir = rdir / "checkpoint"
    marker = train_dir / "done.shard0"

    # Gated on cfg.resume for the same reason as the entropy stage: with
    # resume=false the upstream stages regenerate, and a marker-only skip here
    # would keep a checkpoint trained on data that no longer exists.
    if bool(cfg.resume) and marker.exists() and (checkpoint_dir / "config.json").exists():
        logger.info("train already done for {} round {}, skipping", args.arm, args.round_index)
        return

    selected = read_jsonl(rdir / "selected" / "selected.jsonl")
    if not selected:
        raise RuntimeError(f"no selected trajectories in {rdir / 'selected'}")

    # Same resolution (and same weights-present check) as the rollout and
    # entropy stages -- one implementation, three consumers.
    from apod.stages.rollout_eval import collect_eos_ids, resolve_model_path

    model_path = resolve_model_path(run_dir, args.arm, args.round_index, cfg)
    logger.info(
        "training {} round {}: {} trajectories, student={}, teacher={}",
        args.arm, args.round_index, len(selected), model_path, cfg.model.teacher_id,
    )

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.experimental.gkd import GKDConfig, GKDTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The SAME terminator set the rollout stage used: a trajectory that
    # stopped on a GenerationConfig-declared EOS (<|endoftext|>) must not get
    # a second, different EOS appended by build_rows.
    eos_ids = collect_eos_ids(tokenizer, model_path)

    tc = cfg.train
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective = (
        world_size * int(tc.per_device_train_batch_size) * int(tc.gradient_accumulation_steps)
    )
    if effective != int(tc.effective_batch):
        raise RuntimeError(
            f"effective batch is {world_size} ranks x "
            f"{tc.per_device_train_batch_size} per-device x "
            f"{tc.gradient_accumulation_steps} accum = {effective}, but "
            f"train.effective_batch declares {tc.effective_batch}. Changing "
            "num_gpus requires retuning gradient_accumulation_steps (or "
            "updating effective_batch deliberately)."
        )
    rows, completion_tokens, tail_truncated = build_rows(
        rdir,
        selected,
        eos_ids=eos_ids,
        append_eos_id=int(tokenizer.eos_token_id),
        max_length=int(tc.max_length),
    )
    dataset = Dataset.from_list(rows)

    student = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    student.config.use_cache = False

    gkd_config = GKDConfig(
        output_dir=str(train_dir / "hf"),
        beta=float(tc.beta),
        lmbda=float(tc.lmbda),
        learning_rate=float(tc.learning_rate),
        lr_scheduler_type=str(tc.lr_scheduler_type),
        warmup_steps=int(tc.warmup_steps),
        num_train_epochs=float(tc.num_train_epochs),
        per_device_train_batch_size=int(tc.per_device_train_batch_size),
        gradient_accumulation_steps=int(tc.gradient_accumulation_steps),
        max_grad_norm=float(tc.max_grad_norm),
        weight_decay=float(tc.weight_decay),
        bf16=bool(tc.bf16),
        gradient_checkpointing=bool(tc.gradient_checkpointing),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=bool(tc.use_liger_kernel),
        max_length=int(tc.max_length),
        logging_steps=int(tc.logging_steps),
        save_strategy=str(tc.save_strategy),
        seed=int(tc.seed),
        teacher_model_init_kwargs={"dtype": str(tc.teacher_dtype)},
        report_to=[],
    )

    start = time.perf_counter()
    trainer = GKDTrainer(
        model=student,
        teacher_model=str(cfg.model.teacher_id),
        args=gkd_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    # TRL passes 3D [B, T, H] hidden states into LigerFusedLinearJSDLoss, but
    # liger chunks over dim 0 (it expects [rows, H]); with per-device batch 1
    # that is ONE chunk holding the whole sequence, so the full-vocab logits
    # materialize in fp32 anyway (~53 GiB transient at 8192 tokens -> OOM).
    # Flattening to [B*T, H] / [B*T] is mathematically identical (positions
    # are independent, ignore_index=-100 masks pads, same normalization set)
    # and restores real 1024-row chunking. Verified: same loss, peak drops
    # from 75.7 GiB to a bounded chunk-sized allocation.
    if getattr(trainer, "use_liger_gkd_loss", False):
        _liger_loss_3d = trainer.liger_loss

        def _liger_loss_flat(
            *, student_input, student_weight, teacher_input, teacher_weight,
            true_labels, student_bias=None, teacher_bias=None,
        ):
            return _liger_loss_3d(
                student_input=student_input.reshape(-1, student_input.shape[-1]),
                student_weight=student_weight,
                teacher_input=teacher_input.reshape(-1, teacher_input.shape[-1]),
                teacher_weight=teacher_weight,
                true_labels=true_labels.reshape(-1),
                student_bias=student_bias,
                teacher_bias=teacher_bias,
            )

        trainer.liger_loss = _liger_loss_flat

    # TRL guards the teacher with torch.no_grad() but leaves requires_grad
    # True; freeze explicitly so the frozen-teacher guarantee is structural,
    # not just behavioral.
    trainer.teacher_model.eval()
    for p in trainer.teacher_model.parameters():
        p.requires_grad_(False)
    trainer.train()
    wall_clock = time.perf_counter() - start

    # Under DDP only rank 0 writes artifacts; unwrap_model strips the DDP
    # wrapper so the checkpoint reloads without ``module.`` prefixes.
    trainer.accelerator.wait_for_everyone()
    if not trainer.is_world_process_zero():
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    unwrapped.config.use_cache = True
    unwrapped.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    log_history = list(trainer.state.log_history)
    write_jsonl(train_dir / "log_history.jsonl", log_history)
    losses = [entry["loss"] for entry in log_history if "loss" in entry]
    summary = {
        "num_trajectories": len(rows),
        "tokens_trained": completion_tokens,
        "tail_truncated_rows": tail_truncated,
        "train_loss_mean": float(np.mean(losses)) if losses else None,
        "train_loss_final": float(losses[-1]) if losses else None,
        "wall_clock_s": wall_clock,
    }
    (train_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    marker.touch()
    logger.info(
        "trained {} trajectories ({} completion tokens) in {:.1f}s: "
        "loss mean {:.4f} -> final {:.4f}; checkpoint at {}",
        summary["num_trajectories"], summary["tokens_trained"], wall_clock,
        summary["train_loss_mean"] if summary["train_loss_mean"] is not None else float("nan"),
        summary["train_loss_final"] if summary["train_loss_final"] is not None else float("nan"),
        checkpoint_dir,
    )


if __name__ == "__main__":
    main()
