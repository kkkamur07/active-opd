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

Every optimizer step is logged (logging_steps=1) with loss / grad_norm / lr
plus batch diagnostics (``DiagGKDTrainer``): top-16 overlap ratio,
overlap-token advantage, mean |H_S - H_T|, response tokens and cap-hit
fraction of the step's trajectories. The rows land in train/log_history.jsonl
and, through ``apod.tracking``, in the arm's W&B run at the global training
step ``--global-step-offset + trainer step`` (the caller passes the number of
training steps the arm has already taken across earlier refreshes).

    python -m apod.stages.train --run-dir D --arm A --round R [--global-step-offset S]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf
from trl.experimental.gkd import GKDConfig, GKDTrainer
from trl.experimental.gkd import gkd_trainer as _gkd_module
from trl.experimental.utils import DataCollatorForChatML

from apod import tracking
from apod.datasets.io import read_jsonl, write_jsonl
from apod.paths import round_dir
from apod.stages.common import parse_stage_args, stage_parser

K_OVERLAP = 16  # Rethinking OPD's trained k for Eq. 6-7; the same constant as scripts/oracle_kl.py


def parse_args() -> argparse.Namespace:
    parser = stage_parser(description=__doc__, needs_shards=False)
    parser.add_argument(
        "--global-step-offset", type=int, default=0,
        help="training steps the arm took before this launch; per-step W&B rows "
             "are logged at offset + trainer step so refreshes share one x-axis",
    )
    return parse_stage_args(parser)


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
    ``completion_mask`` (0 over the prompt, 1 over the response), ``prompt``
    (text — the collator requires the key even pre-tokenized) and
    ``truncated`` (the npz cap-hit flag, logged per step as cap_hit_frac).
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
            # "length" feeds group_by_length: the collator pads to the batch
            # max, and random batching wastes a measured 9% of tokens on pads.
            rows.append(
                {
                    "input_ids": ids,
                    "completion_mask": mask,
                    "prompt": "",
                    "length": len(ids),
                    "truncated": bool(truncated[j]),
                }
            )
            completion_tokens += len(ids) - prompt_length
    return rows, completion_tokens, tail_truncated


class CollatorWithTruncated:
    """TRL's ChatML collator plus the rows' ``truncated`` flags. The collator
    emits a fixed key set, so the flag has to be re-attached here for the
    per-step cap-hit fraction; the model never sees the extra key (the GKD
    loss calls it with explicit input_ids/attention_mask)."""

    def __init__(self, tokenizer, max_length: int):
        self.inner = DataCollatorForChatML(tokenizer=tokenizer, max_length=max_length)

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        batch = self.inner(examples)
        batch["truncated"] = torch.tensor([bool(e["truncated"]) for e in examples])
        return batch


@torch.no_grad()
def batch_diagnostics(
    student_hidden: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk: int,
    student_bias: torch.Tensor | None = None,
    teacher_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Token sums of the Rethinking-OPD head diagnostics over the labelled
    (response) rows of a flattened ``[rows, H]`` batch, from a chunked
    lm_head pass -- the SAME hidden states the Liger fused loss consumes.

    Returns a float64 tensor ``[overlap_sum, adv_sum, adv_n, abs_dh_sum,
    tokens]``: per token |top16_S cap top16_T|, the teacher's advantage on the
    intersection with both distributions renormalized over it (counted only
    where the intersection is non-empty: ``adv_n``), and |H_S - H_T|. The
    per-token math is scripts/oracle_kl.py's ``_exact_scores`` verbatim (fp32
    log-softmax over bf16 logits); the means are the sums over ``tokens``
    (and ``K_OVERLAP * tokens`` for the ratio), i.e. token-weighted over the
    batch rather than per trajectory.

    Memory: only ``chunk`` rows of full-vocab fp32 log-probs are alive at once
    (student, teacher and one product buffer: 3 x chunk x vocab x 4 bytes,
    ~2.9 GiB at chunk 1024 and 248k vocab), freed chunk by chunk.
    """
    rows = (labels != -100).nonzero(as_tuple=False).squeeze(-1)
    sums = torch.zeros(5, dtype=torch.float64, device=student_hidden.device)
    sums[4] = rows.numel()
    for lo in range(0, rows.numel(), chunk):
        idx = rows[lo : lo + chunk]
        lp_s = torch.log_softmax(F.linear(student_hidden[idx], student_weight, student_bias).float(), -1)
        p = lp_s.exp()
        h_s = -p.mul_(lp_s).sum(-1)
        lp_t = torch.log_softmax(F.linear(teacher_hidden[idx], teacher_weight, teacher_bias).float(), -1)
        p = lp_t.exp()
        h_t = -p.mul_(lp_t).sum(-1)
        del p
        sums[3] += (h_s - h_t).abs().sum()
        # Eq. 6-7: intersection of the two top-16 id sets; teacher advantage
        # with BOTH distributions renormalized over the intersection.
        sv, si = lp_s.topk(K_OVERLAP, dim=-1)
        ti = lp_t.topk(K_OVERLAP, dim=-1).indices
        in_both = (si.unsqueeze(-1) == ti.unsqueeze(-2)).any(-1)
        isz = in_both.sum(-1)
        t_at_s = lp_t.gather(-1, si)
        del lp_s, lp_t
        lp_bar = sv - sv.masked_fill(~in_both, float("-inf")).logsumexp(-1, keepdim=True)
        lq_bar = t_at_s - t_at_s.masked_fill(~in_both, float("-inf")).logsumexp(-1, keepdim=True)
        term = (lp_bar.exp() * (lq_bar - lp_bar)).masked_fill(~in_both, 0.0)
        has = isz > 0
        sums[0] += isz.sum()
        sums[1] += (term.sum(-1) / isz.clamp(min=1))[has].sum()
        sums[2] += has.sum()
    return sums


def _param_block(name: str) -> str:
    """Coarse block of a parameter name, for the bf16-rounding breakdown."""
    if "embed_tokens" in name:
        return "embeddings"
    if "lm_head" in name:
        return "lm_head"
    if ".mlp." in name:
        return "mlp"
    if "attn" in name:  # self_attn (full attention) and linear_attn (Gated DeltaNet)
        return "attention"
    return "other"  # norms and the GDN A_log / dt_bias vectors


@torch.no_grad()
def adam_bf16_rounded(optimizer, names: list[str], lr: float, *, chunk: int = 1 << 24) -> dict[str, float]:
    """Fraction of trainable elements whose Adam update this step rounds to
    zero in bf16, from the optimizer state right after ``optimizer.step()``.

    The student trains in pure bf16 (no fp32 master weights), so an update
    smaller than half an ulp of the weight it is added to is lost: with
    8 significand bits, half-ulp(x) = 2^(floor(log2|x|) - 8). The update is
    recomputed analytically in fp32 from the post-step moments, which are
    exactly the values AdamW used: lr * m_hat / (sqrt(v_hat) + eps) with the
    current per-param step count for bias correction, plus the decoupled
    weight-decay term when weight_decay > 0. Chunked over each flattened
    tensor (``chunk`` elements, ~64 MiB per fp32 buffer) so nothing
    tensor-sized is allocated. ``names`` is the optimizer's parameter order
    (``_optimizer_param_names``). Returns ``{"bf16_rounded_frac": overall,
    "bf16_rounded_frac_<block>": per _param_block}``.
    """
    rounded: dict[str, float] = defaultdict(float)
    total: dict[str, float] = defaultdict(float)
    i = 0
    for group in optimizer.param_groups:
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        wd = float(group.get("weight_decay", 0.0))
        for p in group["params"]:
            name = names[i]
            i += 1
            state = optimizer.state.get(p)
            if not state or "exp_avg" not in state:
                continue
            t = float(state["step"])
            bc1 = 1.0 - beta1**t
            bc2_sqrt = (1.0 - beta2**t) ** 0.5
            w_flat, m_flat, v_flat = p.view(-1), state["exp_avg"].view(-1), state["exp_avg_sq"].view(-1)
            block = _param_block(name)
            n = w_flat.numel()
            total[block] += n
            for lo in range(0, n, chunk):
                hi = min(lo + chunk, n)
                w = w_flat[lo:hi].float()
                upd = (lr / bc1) * m_flat[lo:hi].float() / (v_flat[lo:hi].float().sqrt() / bc2_sqrt + eps)
                if wd:
                    upd += lr * wd * w
                half_ulp = torch.exp2(torch.floor(torch.log2(w.abs())) - 8)
                rounded[block] += float((upd.abs() < half_ulp).sum())
                del w, upd, half_ulp
    out = {"bf16_rounded_frac": sum(rounded.values()) / max(sum(total.values()), 1.0)}
    for block in sorted(total):
        out[f"bf16_rounded_frac_{block}"] = rounded[block] / max(total[block], 1.0)
    return out


class ContinuedSchedule:
    """Trainer mixin: ONE LR schedule over the whole run, not one per launch.

    A run is trained in passes (one train-stage launch per refresh). With
    ``total_training_steps`` set, every pass builds the scheduler for that
    total (warmup measured against it, e.g. warmup_steps 0.05 -> 5 of 100)
    and advances it by ``global_step_offset`` -- the steps earlier passes
    took -- before its first step, while the trainer's own max_steps stays
    the pass length. Unset (None) keeps the per-pass schedule. Adam state
    continuity is train.persist_optimizer, separate from this.
    """

    total_training_steps: int | None = None
    global_step_offset: int = 0

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is not None or self.total_training_steps is None:
            return super().create_scheduler(num_training_steps, optimizer)
        total = int(self.total_training_steps)
        if self.global_step_offset + num_training_steps > total:
            raise RuntimeError(
                f"this pass runs steps {self.global_step_offset + 1}..{self.global_step_offset + num_training_steps} "
                f"but train.total_training_steps is {total}; the run's schedule would be overrun"
            )
        scheduler = super().create_scheduler(total, optimizer)
        with warnings.catch_warnings():
            # LambdaLR warns about stepping before optimizer.step(); the
            # steps being skipped here were taken by earlier passes.
            warnings.simplefilter("ignore")
            for _ in range(self.global_step_offset):
                scheduler.step()
        return scheduler


class DiagGKDTrainer(ContinuedSchedule, GKDTrainer):
    """GKDTrainer with per-step batch diagnostics and W&B step logging.

    Liger's fused JSD never materialises logits, so ``batch_diagnostics`` runs
    inside the ``liger_loss`` call on the flattened hidden states it is about
    to consume (no_grad, chunked, freed as it goes). Sums accumulate over the
    micro-batches of one optimizer step, are all-reduced across ranks, and
    are merged into the trainer's own ``log`` row (logging_steps=1 -> one
    log_history row per step). The loss, optimizer, seeds and resume paths
    are untouched. Without Liger (eager loss) only tokens and cap-hit are
    logged.
    """

    def __init__(
        self, *args, diag_every: int, diag_chunk: int, global_step_offset: int,
        total_training_steps: int | None, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.diag_every = diag_every
        self.diag_chunk = diag_chunk
        self.global_step_offset = global_step_offset
        self.total_training_steps = total_training_steps
        self._param_names: list[str] | None = None
        # [overlap_sum, adv_sum, adv_n, abs_dh_sum, diag_tokens, tokens, cap_hit, trajectories]
        self._sums = torch.zeros(8, dtype=torch.float64, device=self.args.device)
        self._pending = False
        if self.use_liger_gkd_loss:
            self._liger_loss_3d = self.liger_loss
            self.liger_loss = self._liger_loss_flat
        elif diag_every > 0:
            logger.warning("use_liger_kernel is off: overlap/entropy diagnostics are not computed on the eager loss path")

    def _diag_due(self) -> bool:
        return self.diag_every > 0 and (self.state.global_step + 1) % self.diag_every == 0

    def _liger_loss_flat(
        self, *, student_input, student_weight, teacher_input, teacher_weight,
        true_labels, student_bias=None, teacher_bias=None,
    ):
        # TRL passes 3D [B, T, H] hidden states into LigerFusedLinearJSDLoss,
        # but liger chunks over dim 0 (it expects [rows, H]); with per-device
        # batch 1 that is ONE chunk holding the whole sequence, so the
        # full-vocab logits materialize in fp32 anyway (~53 GiB transient at
        # 8192 tokens -> OOM). Flattening to [B*T, H] / [B*T] is
        # mathematically identical (positions are independent,
        # ignore_index=-100 masks pads, same normalization set) and restores
        # real 1024-row chunking. Verified: same loss, peak drops from 75.7
        # GiB to a bounded chunk-sized allocation.
        student_input = student_input.reshape(-1, student_input.shape[-1])
        teacher_input = teacher_input.reshape(-1, teacher_input.shape[-1])
        true_labels = true_labels.reshape(-1)
        if self._diag_due():
            self._sums[:5] += batch_diagnostics(
                student_input, student_weight, teacher_input, teacher_weight, true_labels,
                chunk=self.diag_chunk, student_bias=student_bias, teacher_bias=teacher_bias,
            )
        return self._liger_loss_3d(
            student_input=student_input,
            student_weight=student_weight,
            teacher_input=teacher_input,
            teacher_weight=teacher_weight,
            true_labels=true_labels,
            student_bias=student_bias,
            teacher_bias=teacher_bias,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        # Response tokens = labelled positions after the causal shift, the
        # exact set the loss normalizes over.
        self._sums[5] += (inputs["labels"][:, 1:] != -100).sum()
        truncated = inputs.get("truncated")
        if truncated is not None:
            self._sums[6] += truncated.sum()
            self._sums[7] += truncated.numel()
        self._pending = True
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if self._pending:
            # Every rank reaches here once per optimizer step (the base
            # trainer's loss gather is already a collective at this point).
            sums = self.accelerator.reduce(self._sums.clone(), reduction="sum").tolist()
            self._sums.zero_()
            self._pending = False
            overlap, adv, adv_n, abs_dh, diag_tokens, tokens, cap_hit, trajectories = sums
            logs["response_tokens"] = int(tokens)
            logs["cap_hit_frac"] = cap_hit / max(trajectories, 1.0)
            if diag_tokens > 0:
                logs["overlap_ratio_top16"] = overlap / (K_OVERLAP * diag_tokens)
                logs["overlap_adv_top16"] = adv / max(adv_n, 1.0)
                logs["abs_entropy_gap"] = abs_dh / diag_tokens
            # Optimizer state is replicated under DDP: one rank's numbers are
            # everyone's. logs["learning_rate"] is the lr this step used
            # (captured before the scheduler advanced).
            if "learning_rate" in logs and self.optimizer is not None and self.is_world_process_zero():
                inner = getattr(self.optimizer, "optimizer", self.optimizer)
                if self._param_names is None:
                    self._param_names = _optimizer_param_names(inner, self.model)
                logs.update(adam_bf16_rounded(inner, self._param_names, float(logs["learning_rate"])))
        super().log(logs, start_time)
        if "loss" in logs and self.is_world_process_zero():
            tracking.log_step(
                self.global_step_offset + self.state.global_step,
                {k: v for k, v in logs.items() if isinstance(v, (int, float))},
            )


def _optimizer_param_names(optimizer, model) -> list[str]:
    """Optimizer param order -> parameter names (any DDP ``module.`` stripped).

    AdamW's ``state_dict`` keys states by POSITION over the concatenated
    param_groups, so a raw dump is only loadable if the groups are rebuilt in
    the exact same order. Name-keying the dump makes the cross-round mapping
    explicit and robust (USER 2026-09-01: "with proper key mapping so that
    run can be restarted at any point")."""
    id_to_name = {id(p): n.removeprefix("module.") for n, p in model.named_parameters()}
    return [id_to_name[id(p)] for group in optimizer.param_groups for p in group["params"]]


def _restore_optimizer_state(optimizer, model, saved: dict) -> int:
    """Load a ``{"names", "state_dict"}`` dump into ``optimizer``, remapping
    positional state keys through param names so the current param order need
    not match the one at save time. Current ``param_groups`` (lr etc.) are
    kept: those are scheduler-driven, not carried over. Returns the number of
    restored per-param states."""
    index_of = {n: i for i, n in enumerate(_optimizer_param_names(optimizer, model))}
    remapped = {
        index_of[saved["names"][int(key)]]: value
        for key, value in saved["state_dict"]["state"].items()
    }
    current = optimizer.state_dict()
    current["state"] = remapped
    optimizer.load_state_dict(current)
    return len(remapped)


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

    # TRL flushes the CUDA allocator cache on EVERY compute_loss (128x per
    # rank per round), forcing ~20 GiB of transients to be re-faulted each
    # micro-step. With >20 GiB of measured headroom the flush protects
    # nothing; no-op it (module-level name, looked up at call time).
    _gkd_module.empty_cache = lambda: None

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
        # >= 1: steps; < 1: fraction of the schedule length (transformers
        # 5.x), which is total_training_steps when that is set.
        warmup_steps=float(tc.warmup_steps),
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
        # e.g. {"min_lr_rate": 0.1} for cosine_with_min_lr; absent -> {}.
        lr_scheduler_kwargs=dict(tc.get("lr_scheduler_kwargs") or {}),
        seed=int(tc.seed),
        teacher_model_init_kwargs={"dtype": str(tc.teacher_dtype)},
        report_to=[],
        # Perf-only (identical objective): batch similar-length sequences so
        # the pad-to-batch-max collator wastes ~0.1% instead of a measured 9%
        # of tokens, and prefetch collated batches off the training loop.
        # transformers 5.x spelling: group_by_length=True became
        # train_sampling_strategy="group_by_length" (the old kwarg is gone
        # from TRL 1.10's config chain and raises TypeError).
        train_sampling_strategy="group_by_length",
        length_column_name="length",
        dataloader_num_workers=2,
        dataloader_prefetch_factor=4,
    )

    tracking.init(cfg, run_dir, args.arm)
    start = time.perf_counter()
    trainer = DiagGKDTrainer(
        model=student,
        teacher_model=str(cfg.model.teacher_id),
        args=gkd_config,
        data_collator=CollatorWithTruncated(tokenizer, int(tc.max_length)),
        train_dataset=dataset,
        processing_class=tokenizer,
        diag_every=int(tc.get("diag_every", 1)),
        diag_chunk=int(tc.get("diag_chunk", 1024)),
        global_step_offset=int(args.global_step_offset),
        total_training_steps=(
            int(tc.total_training_steps) if tc.get("total_training_steps") is not None else None
        ),
    )

    # TRL guards the teacher with torch.no_grad() but leaves requires_grad
    # True; freeze explicitly so the frozen-teacher guarantee is structural,
    # not just behavioral.
    trainer.teacher_model.eval()
    for p in trainer.teacher_model.parameters():
        p.requires_grad_(False)

    # Optimizer continuity across rounds (train.persist_optimizer): restore
    # the previous round's Adam moments + per-param step counts so a round
    # restart is a continuation, not a magnitude-blind t=1 sign step. Weights
    # already come from the same checkpoint, so state and params agree. The
    # scheduler stays round-local (fresh warmup+decay cycle per round).
    persist_optimizer = bool(tc.get("persist_optimizer") or False)
    if persist_optimizer and args.round_index > 0:
        prev = (
            round_dir(run_dir, args.arm, args.round_index - 1)
            / "checkpoint" / "optimizer_state.pt"
        )
        if not prev.exists():
            # Same philosophy as resolve_model_path: silently training with
            # reset moments would be a different experiment, not a failed one.
            raise FileNotFoundError(
                f"train.persist_optimizer is set but {prev} is missing; the "
                "previous round's train stage predates the setting or its "
                "state was pruned"
            )
        saved = torch.load(prev, map_location="cpu", weights_only=True)
        trainer.create_optimizer()  # idempotent; train() reuses it
        inner = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
        restored = _restore_optimizer_state(inner, trainer.model, saved)
        logger.info("optimizer state restored from {} ({} params)", prev, restored)

    trainer.train()
    wall_clock = time.perf_counter() - start

    # Under DDP only rank 0 writes artifacts; unwrap_model strips the DDP
    # wrapper so the checkpoint reloads without ``module.`` prefixes.
    trainer.accelerator.wait_for_everyone()
    if not trainer.is_world_process_zero():
        return

    # Atomic checkpoint: save_pretrained writes file by file and
    # resolve_model_path accepts any *.safetensors, so a crash mid-save would
    # leave a checkpoint the next stage happily loads. Write everything into
    # a sibling tmp dir and rename it into place (one rename on the same
    # filesystem); the done marker still comes last.
    staging = checkpoint_dir.with_name(checkpoint_dir.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    unwrapped.config.use_cache = True
    unwrapped.save_pretrained(staging)
    tokenizer.save_pretrained(staging)
    if persist_optimizer and trainer.optimizer is not None:
        # ~8 GB at 2B params (moments are bf16, matching the param dtype);
        # the driver deletes the previous round's file once consumed.
        inner = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
        torch.save(
            {
                "names": _optimizer_param_names(inner, unwrapped),
                "state_dict": inner.state_dict(),
            },
            staging / "optimizer_state.pt",
        )
        logger.info("optimizer state saved to {}", checkpoint_dir / "optimizer_state.pt")
    if checkpoint_dir.exists():  # a partial checkpoint from a crashed earlier save
        shutil.rmtree(checkpoint_dir)
    os.replace(staging, checkpoint_dir)

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
    tracking.finish()
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
