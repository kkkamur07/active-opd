"""CPU verification of ``train.persist_optimizer`` (apod/stages/train.py).

Exercises the exact save/load helpers the train stage uses
(``_optimizer_param_names`` / ``_restore_optimizer_state``) and the same
``transformers.Trainer`` code path GKDTrainer inherits (``create_optimizer``
before ``train()``, ``accelerator.prepare`` wrapping the optimizer in an
``AcceleratedOptimizer``), on a random 2-layer Qwen2 model. No downloads, no
GPU.

Run:  PYTHONPATH=. .venv/bin/python tests/test_persist_optimizer.py
(pytest-compatible: every ``test_*`` function is a plain assert test).
"""

from __future__ import annotations

import math
import os
import socket
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import torch
from datasets import Dataset
from transformers import Qwen2Config, Qwen2ForCausalLM, Trainer, TrainingArguments

from apod.stages.train import ContinuedSchedule, _optimizer_param_names, _restore_optimizer_state


class ContinuedTrainer(ContinuedSchedule, Trainer):
    """Plain Trainer + the schedule-continuation mixin DiagGKDTrainer uses."""

LR = 1e-3
MIN_LR_RATE = 0.1


# --- harness ----------------------------------------------------------------


def tiny_model(seed: int = 0, dtype=torch.float32) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(cfg).to(dtype)


def batch(seed: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 64, (2, 16), generator=g)
    return {"input_ids": ids, "labels": ids}


def hf_grouped_adamw(model: torch.nn.Module) -> torch.optim.AdamW:
    """Same shape as Trainer.create_optimizer: [decay params, no-decay params]."""
    decay = [p for n, p in model.named_parameters() if "norm" not in n and not n.endswith("bias")]
    no_decay = [p for n, p in model.named_parameters() if "norm" in n or n.endswith("bias")]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.0}, {"params": no_decay, "weight_decay": 0.0}], lr=LR
    )


def steps(model: torch.nn.Module, opt: torch.optim.Optimizer, n: int, seed: int = 1) -> None:
    for _ in range(n):
        opt.zero_grad()
        model(**batch(seed)).loss.backward()
        opt.step()


def dump(optimizer, model, path: Path) -> dict:
    """Byte-for-byte the train.py save (lines ~326-337) plus its load
    (``torch.load(..., map_location="cpu", weights_only=True)``)."""
    torch.save(
        {"names": _optimizer_param_names(optimizer, model), "state_dict": optimizer.state_dict()},
        path,
    )
    return torch.load(path, map_location="cpu", weights_only=True)


def state_by_name(optimizer, model) -> dict[str, dict[str, torch.Tensor]]:
    names = _optimizer_param_names(optimizer, model)
    return {names[i]: s for i, s in optimizer.state_dict()["state"].items()}


def assert_same_state(a: dict, b: dict) -> None:
    assert a.keys() == b.keys(), (a.keys() ^ b.keys())
    for name, sa in a.items():
        sb = b[name]
        assert sa.keys() == sb.keys() == {"step", "exp_avg", "exp_avg_sq"}, name
        for k in sa:
            assert torch.equal(sa[k].cpu(), sb[k].cpu()), f"{name}.{k} differs after restore"


def clone_weights(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    dst.load_state_dict(src.state_dict())


# --- (a) save -> load is identical, keyed by parameter name -----------------


def test_roundtrip_identical_by_name(tmp_path: Path | None = None):
    tmp_path = tmp_path or Path(tempfile.mkdtemp())
    model = tiny_model()
    opt = hf_grouped_adamw(model)
    steps(model, opt, 3)
    saved = dump(opt, model, tmp_path / "optimizer_state.pt")
    assert len(saved["names"]) == sum(1 for _ in model.parameters())
    assert all(not n.startswith("module.") for n in saved["names"])

    fresh = tiny_model(seed=99)
    clone_weights(model, fresh)
    opt2 = hf_grouped_adamw(fresh)
    assert opt2.state_dict()["state"] == {}
    restored = _restore_optimizer_state(opt2, fresh, saved)
    assert restored == len(saved["names"])
    assert_same_state(state_by_name(opt, model), state_by_name(opt2, fresh))
    # Per-param Adam step counter carried over (float32 scalar tensor in torch 2.x).
    assert all(float(s["step"]) == 3.0 for s in opt2.state_dict()["state"].values())
    # lr / param-group hyperparameters come from the NEW optimizer, not the dump.
    assert all(g["lr"] == LR for g in opt2.param_groups)


# --- (b) permuted parameter order on load still maps by name ----------------


def test_permuted_param_order_remaps(tmp_path: Path | None = None):
    tmp_path = tmp_path or Path(tempfile.mkdtemp())
    model = tiny_model()
    opt = hf_grouped_adamw(model)
    steps(model, opt, 3)
    saved = dump(opt, model, tmp_path / "optimizer_state.pt")

    fresh = tiny_model(seed=99)
    clone_weights(model, fresh)
    # Single group, reversed named_parameters order: every position differs
    # from the saved [decay, no_decay] layout.
    opt2 = torch.optim.AdamW([p for _, p in reversed(list(fresh.named_parameters()))], lr=LR)
    new_names = _optimizer_param_names(opt2, fresh)
    assert new_names != saved["names"]
    moved = sum(1 for i, n in enumerate(new_names) if saved["names"][i] != n)
    assert moved >= len(new_names) - 1, "permutation should displace (almost) every position"

    _restore_optimizer_state(opt2, fresh, saved)
    assert_same_state(state_by_name(opt, model), state_by_name(opt2, fresh))
    # And the restored state has the right SHAPE per param (a positional load
    # would have put embed_tokens' [64,32] moments on lm_head etc.).
    for i, name in enumerate(new_names):
        st = opt2.state_dict()["state"][i]
        assert st["exp_avg"].shape == dict(fresh.named_parameters())[name].shape, name

    # A name that does not exist in the current model fails loudly (KeyError),
    # never a silent partial restore.
    bad = {"names": ["not.a.param", *saved["names"][1:]], "state_dict": saved["state_dict"]}
    try:
        _restore_optimizer_state(hf_grouped_adamw(fresh), fresh, bad)
    except KeyError:
        pass
    else:
        raise AssertionError("mismatched names should raise")


# --- (d) DDP: no ``module.`` prefix in the dump, and load through the wrapper


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_ddp_prefix_stripped(tmp_path: Path | None = None):
    tmp_path = tmp_path or Path(tempfile.mkdtemp())
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{_free_port()}", rank=0, world_size=1
    )
    try:
        model = tiny_model()
        ddp = DDP(model)
        assert all(n.startswith("module.") for n, _ in ddp.named_parameters())
        # Trainer builds the optimizer over the UNWRAPPED self.model (same
        # tensors DDP holds), then accelerate wraps the model; mirror that.
        opt = hf_grouped_adamw(model)
        for _ in range(2):
            opt.zero_grad()
            ddp(**batch()).loss.backward()
            opt.step()
        # Names resolved through the DDP wrapper (worst case) carry no prefix.
        assert _optimizer_param_names(opt, ddp) == _optimizer_param_names(opt, model)
        saved = dump(opt, ddp, tmp_path / "optimizer_state.pt")
        assert all(not n.startswith("module.") for n in saved["names"])

        fresh = tiny_model(seed=7)
        clone_weights(model, fresh)
        fresh_ddp = DDP(fresh)
        opt2 = hf_grouped_adamw(fresh)
        # Load handles both: model given as the wrapper ...
        assert _restore_optimizer_state(opt2, fresh_ddp, saved) == len(saved["names"])
        assert_same_state(state_by_name(opt, model), state_by_name(opt2, fresh))
        # ... or as the bare module.
        opt3 = hf_grouped_adamw(fresh)
        assert _restore_optimizer_state(opt3, fresh, saved) == len(saved["names"])
        assert_same_state(state_by_name(opt, model), state_by_name(opt3, fresh))
    finally:
        dist.destroy_process_group()


# --- saved dtype vs optimizer dtype ----------------------------------------


def test_state_dtype_follows_current_params(tmp_path: Path | None = None):
    """torch's Optimizer.load_state_dict casts floating state to the CURRENT
    param dtype/device (step stays as saved), so a bf16 dump restores into a
    bf16 optimizer bit-for-bit and into an fp32 one as an exact upcast."""
    tmp_path = tmp_path or Path(tempfile.mkdtemp())
    model = tiny_model(dtype=torch.bfloat16)
    opt = hf_grouped_adamw(model)
    steps(model, opt, 2)
    saved = dump(opt, model, tmp_path / "optimizer_state.pt")
    assert saved["state_dict"]["state"][0]["exp_avg"].dtype == torch.bfloat16

    same = tiny_model(seed=3, dtype=torch.bfloat16)
    clone_weights(model, same)
    opt_bf16 = hf_grouped_adamw(same)
    _restore_optimizer_state(opt_bf16, same, saved)
    assert_same_state(state_by_name(opt, model), state_by_name(opt_bf16, same))

    up = tiny_model(seed=3, dtype=torch.float32)
    up.load_state_dict({k: v.float() for k, v in model.state_dict().items()})
    opt_fp32 = hf_grouped_adamw(up)
    _restore_optimizer_state(opt_fp32, up, saved)
    for name, st in state_by_name(opt_fp32, up).items():
        ref = state_by_name(opt, model)[name]
        assert st["exp_avg"].dtype == torch.float32
        assert torch.equal(st["exp_avg"], ref["exp_avg"].float()), name
        assert torch.equal(st["step"], ref["step"])


# --- (c) Trainer continuation: 4 steps == 2 + save/load + 2 -----------------


def _dataset() -> Dataset:
    ids = batch()["input_ids"][:1].tolist()  # ONE row, batch 1: every step is the same batch
    return Dataset.from_dict({"input_ids": ids, "labels": ids})


def _args(out: Path, max_steps: int, sched: str, warmup: float) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(out), max_steps=max_steps, per_device_train_batch_size=1,
        learning_rate=LR, lr_scheduler_type=sched, warmup_steps=warmup,
        lr_scheduler_kwargs={"min_lr_rate": MIN_LR_RATE} if sched == "cosine_with_min_lr" else {},
        weight_decay=0.0, max_grad_norm=1.0, logging_steps=1, save_strategy="no",
        report_to=[], use_cpu=True, seed=0, dataloader_num_workers=0,
    )


def _lrs(trainer: Trainer) -> list[float]:
    return [e["learning_rate"] for e in trainer.state.log_history if "learning_rate" in e]


def _train_split(tmp: Path, sched: str, warmup: float, first: int, second: int, total: int | None = None):
    """Pass 0: fresh trainer, ``first`` steps, save weights + optimizer dump
    exactly like train.py. Pass 1: reload weights, ``create_optimizer()``,
    ``_restore_optimizer_state`` on the inner optimizer, train ``second``.
    With ``total`` set, both passes use ContinuedTrainer (one schedule over
    ``total`` steps, pass 1 advanced by ``first`` -- train.py's
    train.total_training_steps + --global-step-offset).
    Returns (model after pass 1, lrs pass 0, lrs pass 1, steps after load)."""
    cls = ContinuedTrainer if total is not None else Trainer
    tr0 = cls(model=tiny_model(), args=_args(tmp / "p0", first, sched, warmup),
              train_dataset=_dataset())
    tr0.total_training_steps, tr0.global_step_offset = total, 0
    tr0.train()
    unwrapped = tr0.accelerator.unwrap_model(tr0.model)
    inner = getattr(tr0.optimizer, "optimizer", tr0.optimizer)
    assert inner is not tr0.optimizer, "accelerate wraps the optimizer; train.py unwraps it"
    ckpt = tmp / "checkpoint"
    unwrapped.save_pretrained(ckpt)
    saved = dump(inner, unwrapped, ckpt / "optimizer_state.pt")

    tr1 = cls(model=Qwen2ForCausalLM.from_pretrained(ckpt),
              args=_args(tmp / "p1", second, sched, warmup), train_dataset=_dataset())
    tr1.total_training_steps, tr1.global_step_offset = total, first
    tr1.create_optimizer()  # idempotent; train() reuses it (transformers 5.15 _prepare_for_training)
    inner1 = getattr(tr1.optimizer, "optimizer", tr1.optimizer)
    assert _restore_optimizer_state(inner1, tr1.model, saved) == len(saved["names"])
    loaded_steps = {float(s["step"]) for s in inner1.state_dict()["state"].values()}
    tr1.train()
    inner1 = getattr(tr1.optimizer, "optimizer", tr1.optimizer)
    assert inner1 is getattr(tr1.optimizer, "optimizer", None) or inner1 is tr1.optimizer
    final_steps = {float(s["step"]) for s in inner1.state_dict()["state"].values()}
    return tr1.accelerator.unwrap_model(tr1.model), _lrs(tr0), _lrs(tr1), loaded_steps, final_steps


def _train_continuous(tmp: Path, sched: str, warmup: float, n: int, total: int | None = None):
    cls = ContinuedTrainer if total is not None else Trainer
    tr = cls(model=tiny_model(), args=_args(tmp / "cont", n, sched, warmup),
             train_dataset=_dataset())
    tr.total_training_steps, tr.global_step_offset = total, 0
    tr.train()
    return tr.accelerator.unwrap_model(tr.model), _lrs(tr)


def _max_param_diff(a: torch.nn.Module, b: torch.nn.Module) -> float:
    sa, sb = a.state_dict(), b.state_dict()
    return max((sa[k].float() - sb[k].float()).abs().max().item() for k in sa)


def test_trainer_continuation_constant_lr(tmp_path: Path | None = None):
    """With the LR schedule continued (constant), 2 + load + 2 == 4."""
    tmp = tmp_path or Path(tempfile.mkdtemp())
    cont, lrs_cont = _train_continuous(tmp, "constant", 0, 4)
    split, lrs0, lrs1, loaded, final = _train_split(tmp, "constant", 0, 2, 2)
    assert lrs_cont == [LR] * 4 and lrs0 == [LR] * 2 and lrs1 == [LR] * 2
    assert loaded == {2.0}, "Adam step counter must resume at 2 after load"
    assert final == {4.0}, "and continue to 4 (bias correction uses t=3,4 not t=1,2)"
    diff = _max_param_diff(cont, split)
    assert diff <= 1e-6, f"continuation drifted from the continuous run: max |dw| = {diff}"
    print(f"continued-schedule max |dw| vs continuous 4-step run: {diff:.3e}")


def test_trainer_restarted_cosine_schedule(tmp_path: Path | None = None):
    """kl50w's actual schedule (warmup 1 + cosine_with_min_lr 0.1) restarts
    per pass: the scheduler is rebuilt inside train() with last_epoch=-1 and
    num_training_steps = this pass's steps. Adam moments + step counter
    continue; the LR does not. Documents exactly how the two runs differ."""
    tmp = tmp_path or Path(tempfile.mkdtemp())
    sched, warm = "cosine_with_min_lr", 1

    def lam(step: int, total: int) -> float:  # transformers.optimization, min_lr_rate variant
        if step < warm:
            return step / max(1, warm)
        progress = (step - warm) / max(1, total - warm)
        return (0.5 * (1 + math.cos(math.pi * progress))) * (1 - MIN_LR_RATE) + MIN_LR_RATE

    cont, lrs_cont = _train_continuous(tmp, sched, warm, 4)
    split, lrs0, lrs1, loaded, final = _train_split(tmp, sched, warm, 2, 2)

    # Logged learning_rate is the lr the step was taken with. warmup_steps=1
    # means lambda(0)=0: the FIRST update of every pass is a zero-lr step
    # (moments and step counter still advance), then straight to peak.
    assert lrs_cont == [round(LR * lam(s, 4), 12) for s in range(4)] or all(
        abs(a - LR * lam(s, 4)) < 1e-12 for s, a in enumerate(lrs_cont)
    ), lrs_cont
    assert lrs0 == [0.0, LR], lrs0
    assert lrs1 == [0.0, LR], f"pass 1 must restart warmup+cosine: {lrs1}"
    assert loaded == {2.0} and final == {4.0}

    # Steps 3-4 ran at lr [0, peak] instead of [0.775 peak, 0.325 peak] -> the
    # weights differ; the continuation is Adam-continuous, not LR-continuous.
    expected_cont_tail = [LR * lam(2, 4), LR * lam(3, 4)]
    assert abs(expected_cont_tail[0] - 0.775 * LR) < 1e-12 and abs(expected_cont_tail[1] - 0.325 * LR) < 1e-12
    diff = _max_param_diff(cont, split)
    assert diff > 1e-6, "restarted schedule should not reproduce the continuous run"
    print(f"restarted-schedule max |dw| vs continuous 4-step run: {diff:.3e} "
          f"(pass-1 lrs {lrs1} vs continuous tail {expected_cont_tail})")


# --- (e) Trainer continuation with ONE schedule: 20 == 10 + save/load + 10 --


def test_trainer_continued_cosine_schedule(tmp_path: Path | None = None):
    """train.total_training_steps + --global-step-offset (ContinuedSchedule):
    warmup 5% of 20 = 1 step, then cosine_with_min_lr 0.1 down to step 20,
    built once per pass for the whole run and advanced past the steps the
    earlier pass took. Same logged LR sequence and bit-exact weights as the
    continuous 20-step run (the mismatch test (d) documents is gone)."""
    tmp = tmp_path or Path(tempfile.mkdtemp())
    sched, warm, total = "cosine_with_min_lr", 0.05, 20
    warm_steps = math.ceil(warm * total)

    def lam(step: int) -> float:  # transformers.optimization, min_lr_rate variant
        if step < warm_steps:
            return step / max(1, warm_steps)
        progress = (step - warm_steps) / max(1, total - warm_steps)
        return (0.5 * (1 + math.cos(math.pi * progress))) * (1 - MIN_LR_RATE) + MIN_LR_RATE

    cont, lrs_cont = _train_continuous(tmp, sched, warm, total, total=total)
    split, lrs0, lrs1, loaded, final = _train_split(tmp, sched, warm, 10, 10, total=total)

    expected = [LR * lam(s) for s in range(total)]
    assert all(abs(a - b) < 1e-15 for a, b in zip(lrs_cont, expected)) and len(lrs_cont) == total, lrs_cont
    assert lrs0 == lrs_cont[:10], (lrs0, lrs_cont[:10])
    assert lrs1 == lrs_cont[10:], f"pass 1 must continue the schedule at step 10: {lrs1} vs {lrs_cont[10:]}"
    assert lrs1[0] < lrs0[-1] < LR, "pass 1 starts on the cosine tail, not at warmup 0 / peak"
    assert loaded == {10.0} and final == {20.0}
    diff = _max_param_diff(cont, split)
    assert diff == 0.0, f"continued schedule must reproduce the continuous run bit-exactly: max |dw| = {diff}"
    print(f"continued-schedule (warmup+cosine, 10+10 vs 20) max |dw| = {diff:.1e}; "
          f"lrs pass 1 = {[round(x / LR, 4) for x in lrs1]} x peak")

    # Overrunning the run's schedule fails loudly instead of extrapolating.
    tr = ContinuedTrainer(model=tiny_model(), args=_args(tmp / "over", 11, sched, warm),
                          train_dataset=_dataset())
    tr.total_training_steps, tr.global_step_offset = total, 10
    try:
        tr.train()
    except RuntimeError as exc:
        assert "total_training_steps" in str(exc), exc
    else:
        raise AssertionError("offset 10 + 11 steps > total 20 must raise")


if __name__ == "__main__":
    import sys

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
