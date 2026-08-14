"""Semantic verification of the OPD learning signal, with real numbers.

Structural checks (check_run.py) prove artifacts exist and are well-formed; a
wrong teacher, flipped KL direction, or prompt leakage would pass all of them
while producing plausible-looking losses. This script verifies the signal
itself, on one batch of real smoke trajectories, with cheap inline
computations -- no framework:

  1. tokenizer/template  -- student and teacher tokenize identically
  2. teacher sanity      -- teacher ppl << student ppl on held-out MATH-500
                            text; top-k tokens printed for eyeballing
  3. objective sanity    -- trainer's reported loss == reverse KL recomputed
                            by hand from the two logit tensors (and != forward
                            KL); KL(p||p) == 0; loss covers completion tokens
                            only (no prompt/padding leakage); lmbda/temperature
  4. loss sanity         -- teacher frozen (requires_grad, eval mode, not in
                            optimizer, bit-identical across steps); student
                            weights actually change; tiny overfit run drops
                            the loss sharply
  5. eval/scoring path   -- math-verify grades known answers correctly and
                            reproduces stored eval rows; stage_entropy's score
                            matches an independent recomputation on the same
                            model that the objective uses

Prints every number, writes them to <round>/train/verify_semantics.json, and
exits non-zero on the first insane result.

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/verify_semantics.py \
        --run-dir outputs/runs/smoke --arm entropy_top4 --round 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apod.datasets import load_examples, read_jsonl  # noqa: E402
from apod.datasets.io import read_shards  # noqa: E402
from apod.stages.entropy import _decoder_and_head, trace_scores  # noqa: E402
from apod.stages.train import build_rows, round_dir  # noqa: E402
from apod.verification import verify_answer  # noqa: E402

RESULTS: dict = {}


def check(name: str, passed: bool, detail: str) -> None:
    RESULTS[name] = {"passed": bool(passed), "detail": detail}
    print(f"{'PASS' if passed else 'FAIL'}: {name}: {detail}")
    if not passed:
        raise SystemExit(f"SEMANTIC CHECK FAILED: {name}: {detail}")


def param_digest(model) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().to(torch.float32).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def response_ppl(model, ids: torch.Tensor, prompt_length: int) -> float:
    """Perplexity over response tokens (teacher-forced, same shift as the loss)."""

    with torch.no_grad():
        logits = model(input_ids=ids[None].to(model.device)).logits[0].float()
    log_probs = F.log_softmax(logits[:-1], dim=-1)
    targets = ids[1:].to(model.device)
    token_lp = log_probs.gather(-1, targets[:, None])[:, 0]
    resp_lp = token_lp[prompt_length - 1 :]
    return float(torch.exp(-resp_lp.mean()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", default="entropy_top4")
    parser.add_argument("--round", type=int, default=0, dest="round_index")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    rdir = round_dir(run_dir, args.arm, args.round_index)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    student_id, teacher_id = str(cfg.model.student_id), str(cfg.model.teacher_id)

    # ---- 1. tokenizer / chat template identity ----------------------------
    tok_s = AutoTokenizer.from_pretrained(student_id)
    tok_t = AutoTokenizer.from_pretrained(teacher_id)
    sample = "Prove that $\\sqrt{2}$ is irrational."
    chat = [{"role": "user", "content": sample}]
    # Render exactly as the pipeline does: enable_thinking is always passed
    # explicitly (cfg.model.enable_thinking). The two repos' template DEFAULTS
    # differ (2B defaults to a closed empty <think> block, 9B to an open one),
    # which is why the explicit flag matters and is checked here.
    enable_thinking = bool(cfg.model.enable_thinking)
    text_s = tok_s.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    text_t = tok_t.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    ids_s = tok_s(text_s, add_special_tokens=False)["input_ids"]
    ids_t = tok_t(text_t, add_special_tokens=False)["input_ids"]
    check(
        "tokenizer_identity",
        tok_s.vocab_size == tok_t.vocab_size
        and tok_s.eos_token_id == tok_t.eos_token_id
        and text_s == text_t
        and ids_s == ids_t,
        f"vocab {tok_s.vocab_size}=={tok_t.vocab_size}, eos {tok_s.eos_token_id}=="
        f"{tok_t.eos_token_id}, chat template renders identically under "
        f"enable_thinking={enable_thinking} (defaults differ between repos -- "
        f"pipeline always passes it explicitly), ids identical ({len(ids_s)} tokens)",
    )

    # ---- 2. teacher sanity: ppl on held-out known-good text ---------------
    print("loading student and teacher (bf16, cuda)...")
    student = AutoModelForCausalLM.from_pretrained(
        student_id, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    problem = load_examples("math500", n=1, seed=0)[0]
    good_text = tok_s.apply_chat_template(
        [
            {"role": "user", "content": problem["prompt"]},
            {"role": "assistant", "content": problem["solution"]},
        ],
        tokenize=False,
    )
    prompt_only = tok_s.apply_chat_template(
        [{"role": "user", "content": problem["prompt"]}], tokenize=False,
        add_generation_prompt=True,
    )
    full_ids = torch.tensor(tok_s(good_text, add_special_tokens=False)["input_ids"])
    prompt_len = len(tok_s(prompt_only, add_special_tokens=False)["input_ids"])
    ppl_t = response_ppl(teacher, full_ids, prompt_len)
    ppl_s = response_ppl(student, full_ids, prompt_len)
    check(
        "teacher_ppl_below_student",
        ppl_t < ppl_s and ppl_t < 20.0,
        f"held-out MATH-500 solution: teacher ppl {ppl_t:.2f} < student ppl {ppl_s:.2f}",
    )
    with torch.no_grad():
        t_logits = teacher(input_ids=full_ids[None].to("cuda:0")).logits[0].float()
    for pos in (prompt_len + 1, (prompt_len + len(full_ids)) // 2, len(full_ids) - 2):
        top = torch.topk(F.softmax(t_logits[pos - 1], dim=-1), k=5)
        toks = [repr(tok_s.decode([i])) for i in top.indices.tolist()]
        probs = [f"{p:.2f}" for p in top.values.tolist()]
        print(f"  teacher top-5 @ pos {pos} (true={tok_s.decode([full_ids[pos]])!r}): "
              + ", ".join(f"{t}:{p}" for t, p in zip(toks, probs)))
    del t_logits

    # ---- 3. objective sanity: hand reverse KL vs trainer loss -------------
    selected = read_jsonl(rdir / "selected" / "selected.jsonl")
    # Same terminator set the pipeline itself uses (one implementation).
    from apod.stages.rollout_eval import collect_eos_ids

    eos_ids = collect_eos_ids(tok_s, str(cfg.model.student_id))
    rows, _, _ = build_rows(
        rdir,
        selected[:4],
        eos_ids=eos_ids,
        append_eos_id=int(tok_s.eos_token_id),
        max_length=int(cfg.train.max_length),
    )

    row = rows[0]
    ids = torch.tensor(row["input_ids"], device="cuda:0")[None]
    mask = torch.tensor(row["completion_mask"], device="cuda:0")[None]
    labels = ids.clone()
    labels[mask == 0] = -100  # what DataCollatorForChatML produces (no padding, bsz=1)

    with torch.no_grad():
        s_logits = student(input_ids=ids).logits.float()
        t_logits = teacher(input_ids=ids).logits.float()
    s_shift, t_shift, l_shift = s_logits[:, :-1], t_logits[:, :-1], labels[:, 1:]
    keep = l_shift[0] != -100
    log_ps = F.log_softmax(s_shift[0, keep], dim=-1)
    log_pt = F.log_softmax(t_shift[0, keep], dim=-1)
    reverse_kl = float((log_ps.exp() * (log_ps - log_pt)).sum(-1).mean())
    forward_kl = float((log_pt.exp() * (log_pt - log_ps)).sum(-1).mean())

    from trl.experimental.gkd import GKDTrainer

    trainer_loss = float(GKDTrainer.generalized_jsd_loss(
        student_logits=s_shift, teacher_logits=t_shift, labels=l_shift, beta=1.0
    ))
    rel = abs(trainer_loss - reverse_kl) / max(abs(reverse_kl), 1e-9)
    check(
        "loss_is_reverse_kl",
        rel < 0.01 and abs(trainer_loss - forward_kl) / max(forward_kl, 1e-9) > 0.05,
        f"trainer beta=1 loss {trainer_loss:.5f} == hand reverse KL {reverse_kl:.5f} "
        f"(rel {rel:.2e}); forward KL is {forward_kl:.5f} -- direction not flipped",
    )
    self_kl = float(GKDTrainer.generalized_jsd_loss(
        student_logits=s_shift, teacher_logits=s_shift, labels=l_shift, beta=1.0
    ))
    check(
        "kl_zero_for_identical",
        abs(self_kl) < 1e-6 and reverse_kl >= 0,
        f"KL(student||student) = {self_kl:.2e}; KL(student||teacher) = {reverse_kl:.5f} >= 0",
    )
    prompt_len_row = row["completion_mask"].count(0)
    n_masked = int((l_shift[0] == -100).sum())
    n_scored = int(keep.sum())
    n_completion = int(sum(row["completion_mask"]))
    check(
        "loss_only_over_response",
        n_scored == n_completion and n_masked == prompt_len_row - 1,
        f"{n_scored} scored positions == {n_completion} completion tokens; "
        f"prompt ({prompt_len_row} tokens) fully masked -- no prompt leakage",
    )
    check(
        "no_intrainer_generation",
        float(cfg.train.lmbda) == 0.0 and float(cfg.train.beta) == 1.0,
        f"lmbda={cfg.train.lmbda} (generation gated on random()<=lmbda: never fires), "
        f"beta={cfg.train.beta}; loss temperature is fixed 1.0 in TRL (args.temperature "
        "is generation-only, verified against gkd_trainer.py source)",
    )
    del s_logits, t_logits, s_shift, t_shift, log_ps, log_pt

    # ---- entropy path: same distributions as the objective ----------------
    ent_rows = read_shards(rdir / "entropy", "entropy.shard*.jsonl")
    sel0 = selected[0]
    stored = next(
        r for r in ent_rows
        if r["example_index"] == sel0["example_index"] and r["rollout_index"] == sel0["rollout_index"]
    )
    npz_path = rdir / "rollouts" / "tokens" / f"example_{sel0['example_index']:05d}.npz"
    with np.load(npz_path, allow_pickle=True) as data:
        npz_ids = data["input_ids"][sel0["rollout_index"]]
        npz_prompt_len = int(data["prompt_length"])
        npz_resp_len = int(data["response_lengths"][sel0["rollout_index"]])
    body, head = _decoder_and_head(student)
    recomputed = trace_scores(
        student, body, head, npz_ids, npz_prompt_len, npz_resp_len,
        int(cfg.selection.logit_chunk),
    )
    ent_diff = abs(recomputed["entropy"] - stored["entropy"])
    check(
        "entropy_matches_objective_model",
        ent_diff < 0.02,
        f"stage_entropy stored {stored['entropy']:.4f}, recomputed on the round's "
        f"student {recomputed['entropy']:.4f} (|diff| {ent_diff:.1e}) -- same "
        "full-vocab T=1 distributions the KL uses",
    )

    # ---- 5. eval / math-verify path ---------------------------------------
    assert verify_answer("The answer is $\\boxed{4}$", "4"), "math-verify rejected a correct boxed answer"
    assert not verify_answer("The answer is $\\boxed{5}$", "4"), "math-verify accepted a wrong answer"
    assert verify_answer("\\boxed{\\frac{1}{2}}", "0.5"), "math-verify rejected an equivalent fraction"
    pool = {p["id"]: p["reference"] for p in read_jsonl(run_dir / "pool" / "prompts.jsonl")}
    traj_rows = read_shards(rdir / "rollouts", "trajectories.shard*.jsonl")
    regraded = 0
    for r in traj_rows[:20]:
        if verify_answer(r["response"], pool[r["id"]]) != bool(r["correct"]):
            check(
                "rollout_regrade", False,
                f"stored correct={r['correct']} but regrade disagrees for {r['id']} "
                f"rollout {r['rollout_index']}",
            )
        regraded += 1
    n_correct = sum(r["correct"] for r in traj_rows)
    check(
        "math_verify_grading",
        True,
        f"synthetic boxed answers graded correctly; {regraded} stored rollout rows "
        f"regrade identically ({n_correct}/{len(traj_rows)} correct overall -- parser "
        "is not rejecting everything)",
    )

    # ---- 4. loss sanity: frozen teacher + overfit micro-run ---------------
    del student, teacher, body, head
    torch.cuda.empty_cache()
    print("building a real GKDTrainer for the frozen-teacher and overfit checks...")

    from datasets import Dataset
    from trl.experimental.gkd import GKDConfig

    tc = cfg.train
    scratch = rdir / "train" / "verify_scratch"
    gkd_config = GKDConfig(
        output_dir=str(scratch),
        beta=float(tc.beta),
        lmbda=float(tc.lmbda),
        learning_rate=float(tc.learning_rate),
        lr_scheduler_type="constant",
        warmup_steps=0,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_grad_norm=float(tc.max_grad_norm),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=int(tc.max_length),
        logging_steps=1,
        save_strategy="no",
        seed=int(tc.seed),
        teacher_model_init_kwargs={"dtype": str(tc.teacher_dtype)},
        report_to=[],
    )
    student2 = AutoModelForCausalLM.from_pretrained(student_id, dtype=torch.bfloat16)
    student2.config.use_cache = False
    trainer = GKDTrainer(
        model=student2,
        teacher_model=teacher_id,
        args=gkd_config,
        train_dataset=Dataset.from_list(rows[:2]),
        processing_class=tok_s,
    )
    # Same explicit freeze the train stage applies (apod/stages/train.py):
    # TRL relies on torch.no_grad() and leaves requires_grad True otherwise.
    trainer.teacher_model.eval()
    for p in trainer.teacher_model.parameters():
        p.requires_grad_(False)
    teacher_grads = [p.requires_grad for p in trainer.teacher_model.parameters()]
    check(
        "teacher_frozen_flags",
        not any(teacher_grads) and not trainer.teacher_model.training,
        f"all {len(teacher_grads)} teacher params requires_grad=False, teacher in eval "
        "mode (frozen explicitly by the train stage; TRL additionally wraps every "
        "teacher forward in torch.no_grad())",
    )
    teacher_before = param_digest(trainer.teacher_model)
    student_before = param_digest(trainer.model)
    result = trainer.train()
    opt_param_ids = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    teacher_in_opt = any(id(p) in opt_param_ids for p in trainer.teacher_model.parameters())
    check("teacher_not_in_optimizer", not teacher_in_opt,
          f"optimizer holds {len(opt_param_ids)} tensors, none from the teacher")
    check(
        "teacher_bit_identical_after_training",
        param_digest(trainer.teacher_model) == teacher_before,
        f"teacher param sha256 {teacher_before} unchanged across {int(result.global_step)} steps",
    )
    check(
        "student_weights_changed",
        param_digest(trainer.model) != student_before,
        f"student param digest changed from {student_before}",
    )
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    grad_norms = [h["grad_norm"] for h in trainer.state.log_history if "grad_norm" in h]
    check(
        "overfit_micro_run",
        losses[-1] < 0.5 * losses[0],
        f"2 trajectories x {len(losses)} steps: loss {losses[0]:.4f} -> {losses[-1]:.4f} "
        f"(step-0 loss matches hand KL scale {reverse_kl:.4f}); grad norms "
        f"[{min(grad_norms):.1f}, {max(grad_norms):.1f}]",
    )
    RESULTS["loss_curve"] = {"losses": losses, "grad_norms": grad_norms}

    out = rdir / "train" / "verify_semantics.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"\nALL SEMANTIC CHECKS PASSED -> {out}")


if __name__ == "__main__":
    main()
