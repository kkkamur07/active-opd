"""Oracle-KL diagnostic (todo: run ONCE, standalone; USER 2026-08-15: run
after the apod run finishes).

Computes BOTH divergence directions per stored round-0 rollout (USER
2026-08-15: "do the oracle experiment with both reverse and forward KL so
that we can genuinely understand if there is a selection effect"):
  - reverse KL(student || teacher): the exact quantity the beta=1.0 GKD
    objective minimizes -- upweights positions where the student is
    confidently wrong under the teacher; the training-oracle score.
  - forward KL(teacher || student): mass-covering direction -- upweights
    positions where the teacher has mass the student lacks (coverage gaps).
Both come from the same two forward passes, so forward KL is free.

Also per trajectory, from the same fp32 log-softmaxes at negligible extra
cost: top-16 head agreement (Rethinking OPD, arXiv 2604.13016, Eq. 6-7) --
the overlap ratio of the two models' top-16 id sets, and the overlap-token
advantage with both distributions renormalized over the intersection.
Analysis-only, never a loss term; k is fixed at 16, the paper's trained k
(USER 2026-08-31: no k sweep).

The attribution question for the entropy-vs-random null: if entropy-top4 and
KL-top4 pick nearly the same trajectories, the selection PREMISE is weak; if
they pick differently, mean entropy is the wrong proxy and a better statistic
is the fix. Disagreement BETWEEN the two KL directions additionally tells us
whether any selection effect depends on the divergence direction.

Per trajectory it writes: mean reverse KL over response tokens, plus the
stored entropy/logprob joined from the entropy stage. Analysis (per-prompt
rank correlation, top-4 overlap, KL range percentiles) runs with --analyze
after all shards finish.

Run (per GPU shard):
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/oracle_kl.py \
      --run-dir outputs/runs/apod --arm entropy_top4 --round 0 \
      --shard 0 --num-shards 2
  CUDA_VISIBLE_DEVICES=1 ... --shard 1 --num-shards 2
Then:
  uv run python scripts/oracle_kl.py --run-dir outputs/runs/apod \
      --arm entropy_top4 --round 0 --analyze

--estimator mc (perf_review finding 4, validation step): the single-sample
Monte-Carlo reverse KL on the SAMPLED tokens, mean_t log pi_S(y_t) -
log pi_T(y_t), from the same two HF forwards but gathering only the taken
token's log-prob (logit - logsumexp; no full-vocab log-softmax, entropies or
top-16 sets). Rows go to oracle_kl_mc.shard{K}.jsonl as ``rkl_mc``; the exact
oracle_kl.shard{K}.jsonl rows are never touched and selection still reads
only those. The HF path was chosen over vLLM prompt_logprobs because it is
already wired here (loader, body/head split, same-length batching, resume)
and needs no engine; the cheaper vLLM route only pays off once this
estimator is validated as a selection statistic. --validate-mc (no GPU)
joins both files and reports per-prompt Spearman and tertile agreement
(kl_high/mid/low membership, the write_selection rule) of rkl_mc against
the exact mean_reverse_kl.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from apod import paths

STUDENT_ID = "Qwen/Qwen3.5-2B"
TEACHER_ID = "Qwen/Qwen3.5-9B"
POSITION_CHUNK = 1024  # fp32 log-softmax over 248k vocab, 1024 positions ~ 1 GiB
K_OVERLAP = 16  # Rethinking-OPD's trained k for Eq. 6-7; fixed, no sweep (USER 2026-08-31)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--arm", default="entropy_top4")
    p.add_argument("--round", type=int, default=0, dest="round_index")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--analyze", action="store_true", help="analyze finished shards, no GPU")
    p.add_argument("--estimator", choices=("exact", "mc"), default="exact",
                   help="exact: full-vocab KLs/entropies/top-16 (the selection statistic); "
                        "mc: sampled-token estimate, written to oracle_kl_mc.shard*.jsonl")
    p.add_argument("--validate-mc", action="store_true",
                   help="no GPU: per-prompt Spearman + tertile agreement of rkl_mc vs exact")
    p.add_argument("--student-path", default=STUDENT_ID,
                   help="student model id or checkpoint dir (re-scoring later rounds)")
    p.add_argument("--tokens-dir", type=Path, default=None,
                   help="override the rollout tokens dir (e.g. teacher trajectories)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="override the output dir (default <round>/oracle)")
    return p.parse_args()


def round_dir(args) -> Path:
    return paths.round_dir(args.run_dir, args.arm, args.round_index)


def oracle_dir(args) -> Path:
    return args.out_dir if args.out_dir is not None else round_dir(args) / "oracle"


def out_path(args, shard: int) -> Path:
    stem = "oracle_kl_mc" if args.estimator == "mc" else "oracle_kl"
    return oracle_dir(args) / f"{stem}.shard{shard}.jsonl"


MAX_BATCH = 4  # same-length rollouts scored per forward; bounds the fp32
               # chunk transients at ~4 GiB per model


def _exact_scores(s_head, t_head, s_hidden, t_hidden, plen: int, total: int) -> dict[str, list[float]]:
    """Full-vocab reverse/forward KL, both entropies and the top-16 agreement
    (Eq. 6-7) at the response positions, from the two decoder outputs."""
    import torch

    b = s_hidden.shape[0]
    device = s_hidden.device
    # Reverse KL at response positions: prediction at t-1 scores the token
    # emitted at t -> [plen-1, total-1).
    rkl_sum = torch.zeros(b, device=device)
    fkl_sum = torch.zeros(b, device=device)
    sent_sum = torch.zeros(b, device=device)
    tent_sum = torch.zeros(b, device=device)
    isz_sum = torch.zeros(b, device=device)
    adv_sum = torch.zeros(b, device=device)
    adv_n = torch.zeros(b, device=device)
    n = 0
    for lo in range(plen - 1, total - 1, POSITION_CHUNK):
        hi = min(lo + POSITION_CHUNK, total - 1)
        lp_s = torch.log_softmax(s_head(s_hidden[:, lo:hi]).float(), dim=-1)
        lp_t = torch.log_softmax(t_head(t_hidden[:, lo:hi]).float(), dim=-1)
        diff = lp_s - lp_t
        rkl_sum += (lp_s.exp() * diff).sum(-1).sum(-1)   # KL(S||T)
        fkl_sum += (lp_t.exp() * -diff).sum(-1).sum(-1)  # KL(T||S)
        # Mean token entropies of both models on the same positions (student
        # entropy cross-checks the entropy stage; teacher entropy is new
        # signal).
        sent_sum += -(lp_s.exp() * lp_s).sum(-1).sum(-1)
        tent_sum += -(lp_t.exp() * lp_t).sum(-1).sum(-1)
        n += hi - lo
        # Top-16 head agreement (Eq. 6-7): |A cap B| where A/B are the
        # student's/teacher's top-16 ids, and the teacher's advantage on the
        # intersection with BOTH distributions renormalized over it. A
        # position with an empty intersection contributes to the ratio but
        # not the mean advantage.
        sv, si = lp_s.topk(K_OVERLAP, dim=-1)
        ti = lp_t.topk(K_OVERLAP, dim=-1).indices
        in_both = (si.unsqueeze(-1) == ti.unsqueeze(-2)).any(-1)
        isz = in_both.sum(-1)
        t_at_s = lp_t.gather(-1, si)
        lp_bar = sv - sv.masked_fill(~in_both, float("-inf")).logsumexp(-1, keepdim=True)
        lq_bar = t_at_s - t_at_s.masked_fill(~in_both, float("-inf")).logsumexp(-1, keepdim=True)
        term = (lp_bar.exp() * (lq_bar - lp_bar)).masked_fill(~in_both, 0.0)
        valid = isz > 0
        isz_sum += isz.sum(-1).float()
        adv_sum += (term.sum(-1) / isz.clamp(min=1)).masked_fill(~valid, 0.0).sum(-1)
        adv_n += valid.sum(-1).float()
    return {
        "mean_reverse_kl": [float(rkl_sum[r]) / max(n, 1) for r in range(b)],
        "mean_forward_kl": [float(fkl_sum[r]) / max(n, 1) for r in range(b)],
        "student_entropy": [float(sent_sum[r]) / max(n, 1) for r in range(b)],
        "teacher_entropy": [float(tent_sum[r]) / max(n, 1) for r in range(b)],
        "overlap_ratio_top16": [float(isz_sum[r]) / max(n * K_OVERLAP, 1) for r in range(b)],
        "overlap_adv_top16": [float(adv_sum[r]) / max(float(adv_n[r]), 1.0) for r in range(b)],
    }


def _mc_scores(s_head, t_head, s_hidden, t_hidden, ids, plen: int, total: int) -> dict[str, list[float]]:
    """Single-sample reverse-KL estimate on the SAMPLED tokens: mean over
    response positions of log pi_S(y_t) - log pi_T(y_t). Only the taken
    token's log-prob is gathered (logit minus logsumexp), so no full-vocab
    log-softmax is materialised."""
    import torch

    b = s_hidden.shape[0]
    diff_sum = torch.zeros(b, device=s_hidden.device)
    n = 0
    for lo in range(plen - 1, total - 1, POSITION_CHUNK):
        hi = min(lo + POSITION_CHUNK, total - 1)
        target = ids[:, lo + 1 : hi + 1].unsqueeze(-1)  # token emitted at t
        logits_s = s_head(s_hidden[:, lo:hi]).float()
        logits_t = t_head(t_hidden[:, lo:hi]).float()
        lp_s = logits_s.gather(-1, target).squeeze(-1) - logits_s.logsumexp(-1)
        lp_t = logits_t.gather(-1, target).squeeze(-1) - logits_t.logsumexp(-1)
        diff_sum += (lp_s - lp_t).sum(-1)
        n += hi - lo
    return {"rkl_mc": [float(diff_sum[r]) / max(n, 1) for r in range(b)]}


def compute(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    from apod.stages.entropy import _decoder_and_head

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens_dir = args.tokens_dir if args.tokens_dir is not None else round_dir(args) / "rollouts" / "tokens"
    npzs = sorted(tokens_dir.glob("example_*.npz"))
    npzs = [p for i, p in enumerate(npzs) if i % args.num_shards == args.shard]

    out = out_path(args, args.shard)
    out.parent.mkdir(parents=True, exist_ok=True)
    # exact: pre-Eq.6-7 rows get re-scored; mc: its own file, its own key.
    done_key = "rkl_mc" if args.estimator == "mc" else "overlap_ratio_top16"
    done = set()
    if out.exists():
        with out.open() as f:
            done = {
                (r["example_index"], r["rollout_index"])
                for r in map(json.loads, f)
                if done_key in r
            }

    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID, dtype=torch.bfloat16, device_map=device
    ).eval()
    student = AutoModelForCausalLM.from_pretrained(
        args.student_path, dtype=torch.bfloat16, device_map=device
    ).eval()
    # Split body/head so the full [T, 248k] logit tensor (8 GiB bf16 at 16k
    # tokens) never materialises: run each decoder once, apply the head per
    # position chunk. Same pattern as the entropy stage; identical numbers.
    t_body, t_head = _decoder_and_head(teacher)
    s_body, s_head = _decoder_and_head(student)

    with out.open("a") as f:
        for npz_path in npzs:
            example_index = int(npz_path.stem.split("_")[1])
            with np.load(npz_path, allow_pickle=True) as d:
                ids_all = d["input_ids"]
                rlens = d["response_lengths"]
                plen = int(d["prompt_length"])
                truncated = d["truncated"]
            # Rollouts of one example share the prompt, so same TOTAL length
            # means identical shape: batch those without any padding or mask.
            by_length: dict[int, list[int]] = defaultdict(list)
            for k in range(ids_all.shape[0]):
                if (example_index, k) not in done:
                    by_length[plen + int(rlens[k])].append(k)
            for total, ks in sorted(by_length.items()):
                for group_start in range(0, len(ks), MAX_BATCH):
                    group = ks[group_start : group_start + MAX_BATCH]
                    rlen = total - plen
                    ids = torch.as_tensor(
                        np.asarray(ids_all[group, :total], dtype=np.int64), device=device
                    )
                    with torch.no_grad():
                        t_hidden = t_body(input_ids=ids, use_cache=False)
                        t_hidden = t_hidden[0] if isinstance(t_hidden, tuple) else t_hidden.last_hidden_state
                        s_hidden = s_body(input_ids=ids, use_cache=False)
                        s_hidden = s_hidden[0] if isinstance(s_hidden, tuple) else s_hidden.last_hidden_state
                        if args.estimator == "mc":
                            scores = _mc_scores(s_head, t_head, s_hidden, t_hidden, ids, plen, total)
                        else:
                            scores = _exact_scores(s_head, t_head, s_hidden, t_hidden, plen, total)
                    for row, k in enumerate(group):
                        f.write(
                            json.dumps(
                                {
                                    "example_index": example_index,
                                    "rollout_index": k,
                                    **{key: values[row] for key, values in scores.items()},
                                    "response_length": rlen,
                                    "truncated": bool(truncated[k]),
                                }
                            )
                            + "\n"
                        )
                    f.flush()
            print(f"example {example_index} done", flush=True)


def _spearman(x, y) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _last_rows(directory: Path, pattern: str, key: str) -> dict[tuple[int, int], dict]:
    """Rows of all shards keyed by trajectory, keeping the LAST row per
    trajectory that carries ``key``: a resume after fields were added
    re-scores old rows and APPENDS, leaving both versions in the file."""
    rows: dict[tuple[int, int], dict] = {}
    for shard_file in sorted(directory.glob(pattern)):
        with shard_file.open() as f:
            for r in map(json.loads, f):
                if key in r:
                    rows[(r["example_index"], r["rollout_index"])] = r
    return rows


def validate_mc(args) -> None:
    """Is rkl_mc a usable stand-in for the exact statistic? Per prompt: rank
    agreement of the two scores over its rollouts, and whether they put each
    rollout in the same tertile (the kl_high/kl_mid/kl_low rule of
    scripts/bucket_experiment.py write_selection: sort desc, 3 equal
    slices)."""
    odir = oracle_dir(args)
    exact = _last_rows(odir, "oracle_kl.shard*.jsonl", "mean_reverse_kl")
    mc = _last_rows(odir, "oracle_kl_mc.shard*.jsonl", "rkl_mc")
    keys = sorted(set(exact) & set(mc))
    print(f"validate-mc: {len(exact)} exact rows, {len(mc)} mc rows, {len(keys)} joined ({odir})")
    if not keys:
        return
    x_all = np.array([exact[k]["mean_reverse_kl"] for k in keys])
    m_all = np.array([mc[k]["rkl_mc"] for k in keys])
    print(
        f"pooled: Spearman {_spearman(x_all, m_all):+.3f}  "
        f"exact mean {x_all.mean():.4f}  mc mean {m_all.mean():.4f}  "
        f"(mc - exact) mean {(m_all - x_all).mean():+.4f} sd {(m_all - x_all).std():.4f}"
    )
    by_example: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for k in keys:
        by_example[k[0]].append(k)
    rhos, agree, per_bucket = [], [], {b: [] for b in ("high", "mid", "low")}
    for ks in by_example.values():
        if len(ks) < 3:
            continue
        x = [exact[k]["mean_reverse_kl"] for k in ks]
        m = [mc[k]["rkl_mc"] for k in ks]
        rho = _spearman(x, m)
        if not np.isnan(rho):
            rhos.append(rho)
        size = len(ks) // 3  # 12 rollouts -> 4 per tertile; a remainder is unbucketed
        order_x = np.argsort(x)[::-1][: 3 * size]
        order_m = np.argsort(m)[::-1][: 3 * size]
        bucket_x = {int(i): rank // size for rank, i in enumerate(order_x)}
        bucket_m = {int(i): rank // size for rank, i in enumerate(order_m)}
        # A trajectory the mc ordering leaves in the remainder is a disagreement.
        agree.append(np.mean([bucket_x[i] == bucket_m.get(i, -1) for i in bucket_x]))
        for t, name in enumerate(("high", "mid", "low")):
            sx = {i for i, b in bucket_x.items() if b == t}
            sm = {i for i, b in bucket_m.items() if b == t}
            per_bucket[name].append(len(sx & sm) / size)
    print(
        f"per-prompt (n={len(agree)}): Spearman mean {np.mean(rhos):+.3f} "
        f"median {np.median(rhos):+.3f}  |  same-tertile {np.mean(agree):.3f} (random 0.333)  |  "
        + "  ".join(f"{name} overlap {np.mean(v):.3f}" for name, v in per_bucket.items())
    )
    for label, flag in (("cap-hit", True), ("finished", False)):
        sel = [k for k in keys if bool(mc[k]["truncated"]) == flag]
        if sel:
            print(
                f"{label} (n={len(sel)}): exact mean "
                f"{np.mean([exact[k]['mean_reverse_kl'] for k in sel]):.4f}  "
                f"mc mean {np.mean([mc[k]['rkl_mc'] for k in sel]):.4f}"
            )


def analyze(args) -> None:
    rdir = round_dir(args)
    oracle = []
    for shard_file in sorted(oracle_dir(args).glob("oracle_kl.shard*.jsonl")):
        with shard_file.open() as f:
            oracle.extend(map(json.loads, f))
    # Keep the LAST row per trajectory: a resume after the Eq. 6-7 fields
    # were added re-scores old rows, leaving both versions in the shard file.
    oracle = list({(r["example_index"], r["rollout_index"]): r for r in oracle}.values())
    entropy = []
    for shard_file in sorted((rdir / "entropy").glob("entropy.shard*.jsonl")):
        with shard_file.open() as f:
            entropy.extend(map(json.loads, f))
    ent = {(r["example_index"], r["rollout_index"]): r["entropy"] for r in entropy}

    by_example: dict[int, list[dict]] = defaultdict(list)
    for r in oracle:
        r["entropy"] = ent.get((r["example_index"], r["rollout_index"]))
        by_example[r["example_index"]].append(r)

    print(f"n trajectories: {len(oracle)}  (examples: {len(by_example)})")
    extra_keys = [
        k for k in ("overlap_ratio_top16", "overlap_adv_top16")
        if oracle and all(k in r for r in oracle)
    ]
    for key in ("mean_reverse_kl", "mean_forward_kl", *extra_keys):
        kls = np.array([r[key] for r in oracle])
        print(
            f"{key} distribution: "
            + "  ".join(f"p{p}={np.percentile(kls, p):.4f}" for p in (5, 25, 50, 75, 95))
            + f"  mean={kls.mean():.4f}"
        )

    # Per-example Spearman rank correlation entropy vs oracle KL, and top-4
    # selection overlap between the two statistics (and expected-random 4/12).
    spearman = _spearman
    pairs = [
        ("entropy", "mean_reverse_kl"),
        ("entropy", "mean_forward_kl"),
        ("mean_reverse_kl", "mean_forward_kl"),
    ]
    pairs += [(k, "mean_reverse_kl") for k in extra_keys]
    rhos = {p: [] for p in pairs}
    overlaps = {p: [] for p in pairs}
    for rows in by_example.values():
        if len(rows) < 2 or any(r["entropy"] is None for r in rows):
            continue
        vals = {
            key: [r[key] for r in rows]
            for key in ("entropy", "mean_reverse_kl", "mean_forward_kl", *extra_keys)
        }
        for a, b in pairs:
            rho = spearman(vals[a], vals[b])
            if not np.isnan(rho):
                rhos[(a, b)].append(rho)
            top_a = set(np.argsort(vals[a])[::-1][:4])
            top_b = set(np.argsort(vals[b])[::-1][:4])
            overlaps[(a, b)].append(len(top_a & top_b))
    for a, b in pairs:
        print(
            f"{a} vs {b}: Spearman mean {np.mean(rhos[(a, b)]):+.3f} "
            f"median {np.median(rhos[(a, b)]):+.3f}  |  top-4 overlap "
            f"{np.mean(overlaps[(a, b)]):.2f}/4 (random 1.33/4)  (n={len(overlaps[(a, b)])})"
        )
    # Termination link (hypothesis #1): do the scores differ for cap-hit vs
    # finished trajectories?
    for key in ("mean_reverse_kl", "mean_forward_kl", "entropy", *extra_keys):
        t = [r[key] for r in oracle if r["truncated"] and r.get(key) is not None]
        fin = [r[key] for r in oracle if not r["truncated"] and r.get(key) is not None]
        if t and fin:
            print(f"{key}: cap-hit mean {np.mean(t):.4f} (n={len(t)})  finished mean {np.mean(fin):.4f} (n={len(fin)})")


def main() -> None:
    args = parse_args()
    if args.analyze:
        analyze(args)
    if args.validate_mc:
        validate_mc(args)
    if not (args.analyze or args.validate_mc):
        compute(args)


if __name__ == "__main__":
    main()
