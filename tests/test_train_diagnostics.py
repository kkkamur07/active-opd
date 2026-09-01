"""``apod.stages.train.batch_diagnostics`` against brute force and against
``scripts/oracle_kl._exact_scores`` (the definition it must reproduce).

Random tensors, CPU only. Runs under pytest or as
``python -m tests.test_train_diagnostics``.
"""

from __future__ import annotations

import torch

from apod.stages.train import K_OVERLAP, adam_bf16_rounded, batch_diagnostics
from scripts.oracle_kl import _exact_scores


def _brute_force(s_hidden, s_weight, t_hidden, t_weight, labels):
    """Full-logit, per-token reference of the same five sums."""
    valid = labels != -100
    # Same precision contract as oracle_kl: the head runs in the model dtype
    # (bf16 logits), the log-softmax in fp32.
    lp_s = torch.log_softmax((s_hidden[valid] @ s_weight.T).float(), -1)
    lp_t = torch.log_softmax((t_hidden[valid] @ t_weight.T).float(), -1)
    h_s = -(lp_s.exp() * lp_s).sum(-1)
    h_t = -(lp_t.exp() * lp_t).sum(-1)
    overlap = adv = adv_n = 0.0
    for i in range(lp_s.shape[0]):
        top_s = set(lp_s[i].topk(K_OVERLAP).indices.tolist())
        top_t = set(lp_t[i].topk(K_OVERLAP).indices.tolist())
        both = sorted(top_s & top_t)
        overlap += len(both)
        if both:
            ids = torch.tensor(both)
            p_bar = torch.softmax(lp_s[i, ids], -1)
            q_bar = torch.softmax(lp_t[i, ids], -1)
            # oracle_kl's per-token advantage: -KL(p_bar || q_bar) divided by
            # the intersection size (an average per overlap token).
            adv += float((p_bar * (q_bar.log() - p_bar.log())).sum()) / len(both)
            adv_n += 1
    return torch.tensor([overlap, adv, adv_n, float((h_s - h_t).abs().sum()), float(valid.sum())], dtype=torch.float64)


def _case(seed: int, rows: int, vocab: int, chunk: int, peaked: bool):
    g = torch.Generator().manual_seed(seed)
    hs, ht = 24, 40
    s_hidden = torch.randn(rows, hs, generator=g).bfloat16()
    t_hidden = torch.randn(rows, ht, generator=g).bfloat16()
    scale = 3.0 if peaked else 0.3  # peaked -> top-16 sets overlap a lot; flat -> rarely
    s_weight = (scale * torch.randn(vocab, hs, generator=g)).bfloat16()
    t_weight = (scale * torch.randn(vocab, ht, generator=g)).bfloat16()
    labels = torch.randint(0, vocab, (rows,), generator=g)
    labels[torch.rand(rows, generator=g) < 0.3] = -100
    got = batch_diagnostics(s_hidden, s_weight, t_hidden, t_weight, labels, chunk=chunk)
    want = _brute_force(s_hidden, s_weight, t_hidden, t_weight, labels)
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-4), f"seed {seed}: {got.tolist()} vs {want.tolist()}"
    assert got[4] == float((labels != -100).sum())
    return got


def test_matches_brute_force_across_chunkings():
    for seed, chunk in ((0, 7), (1, 1), (2, 1000), (3, 16)):
        _case(seed, rows=53, vocab=64, chunk=chunk, peaked=seed % 2 == 0)


def test_chunk_size_is_invisible():
    g = torch.Generator().manual_seed(11)
    s_hidden = torch.randn(90, 16, generator=g).bfloat16()
    t_hidden = torch.randn(90, 20, generator=g).bfloat16()
    s_weight = torch.randn(48, 16, generator=g).bfloat16()
    t_weight = torch.randn(48, 20, generator=g).bfloat16()
    labels = torch.randint(0, 48, (90,), generator=g)
    labels[:10] = -100
    a = batch_diagnostics(s_hidden, s_weight, t_hidden, t_weight, labels, chunk=90)
    b = batch_diagnostics(s_hidden, s_weight, t_hidden, t_weight, labels, chunk=13)
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-6)


def test_empty_intersection_counts_ratio_not_advantage():
    # Disjoint heads: student prefers ids [0, 16), teacher ids [16, 32).
    vocab, rows = 40, 5
    s_weight = torch.zeros(vocab, 1)
    s_weight[:16, 0] = 10.0
    t_weight = torch.zeros(vocab, 1)
    t_weight[16:32, 0] = 10.0
    ones = torch.ones(rows, 1)
    labels = torch.zeros(rows, dtype=torch.long)
    got = batch_diagnostics(ones, s_weight, ones, t_weight, labels, chunk=2)
    assert got[0] == 0 and got[1] == 0 and got[2] == 0 and got[4] == rows


def test_matches_oracle_kl_definition_on_one_trajectory():
    """One trajectory: oracle_kl's per-trajectory means ARE the token means."""
    g = torch.Generator().manual_seed(5)
    hs, ht, vocab, plen, total = 12, 18, 96, 6, 41
    s_head = torch.nn.Linear(hs, vocab, bias=False)
    t_head = torch.nn.Linear(ht, vocab, bias=False)
    with torch.no_grad():
        s_head.weight.copy_(2.0 * torch.randn(vocab, hs, generator=g))
        t_head.weight.copy_(2.0 * torch.randn(vocab, ht, generator=g))
    s_hidden = torch.randn(1, total, hs, generator=g)
    t_hidden = torch.randn(1, total, ht, generator=g)
    ref = _exact_scores(s_head, t_head, s_hidden, t_hidden, plen, total)

    # The trainer's view of the same trajectory: hidden states shifted by one
    # ([:, :-1]) and labels = shifted ids with -100 over the prompt, so the
    # labelled rows are exactly oracle_kl's [plen-1, total-1) window.
    labels = torch.full((total - 1,), -100)
    labels[plen - 1 :] = 1
    got = batch_diagnostics(
        s_hidden[0, :-1], s_head.weight, t_hidden[0, :-1], t_head.weight, labels, chunk=8
    )
    n = total - plen
    assert got[4] == n
    assert abs(got[0] / (K_OVERLAP * n) - ref["overlap_ratio_top16"][0]) < 1e-5
    assert abs(got[1] / max(got[2], 1) - ref["overlap_adv_top16"][0]) < 1e-5
    # |H_S - H_T| per token is a new quantity; its ingredients are oracle_kl's.
    assert got[3] / n >= abs(ref["student_entropy"][0] - ref["teacher_entropy"][0]) - 1e-5


def test_bf16_rounded_fraction():
    """adam_bf16_rounded vs an unchunked fp32 reference on a bf16 AdamW
    state, plus the limits: lr -> 0 loses everything, huge lr loses nothing,
    and the analytic count tracks the empirically unchanged elements."""
    torch.manual_seed(0)
    params = {
        "model.embed_tokens.weight": torch.nn.Parameter(torch.randn(64, 16).bfloat16()),
        "model.layers.0.self_attn.q_proj.weight": torch.nn.Parameter(torch.randn(16, 16).bfloat16()),
        "model.layers.0.mlp.up_proj.weight": torch.nn.Parameter(torch.randn(32, 16).bfloat16()),
        "model.norm.weight": torch.nn.Parameter(torch.ones(16).bfloat16()),
    }
    names = list(params)
    lr = 1e-3
    opt = torch.optim.AdamW(list(params.values()), lr=lr, weight_decay=0.0)
    before = None
    for _ in range(3):
        for p in params.values():
            p.grad = (0.05 * torch.randn_like(p, dtype=torch.float32)).bfloat16()
        before = {n: p.detach().clone() for n, p in params.items()}
        opt.step()

    got = adam_bf16_rounded(opt, names, lr, chunk=5)
    assert got == adam_bf16_rounded(opt, names, lr, chunk=1 << 20), "chunking must be invisible"
    assert set(got) == {"bf16_rounded_frac", "bf16_rounded_frac_embeddings", "bf16_rounded_frac_attention",
                        "bf16_rounded_frac_mlp", "bf16_rounded_frac_other"}, got

    # Unchunked reference, straight from the formula.
    rounded = total = 0
    g = opt.param_groups[0]
    for n, p in params.items():
        st = opt.state[p]
        t = float(st["step"])
        m_hat = st["exp_avg"].float() / (1 - g["betas"][0] ** t)
        v_hat = st["exp_avg_sq"].float() / (1 - g["betas"][1] ** t)
        upd = lr * m_hat / (v_hat.sqrt() + g["eps"])
        half_ulp = 2.0 ** (torch.floor(torch.log2(p.detach().float().abs())) - 8)
        rounded += int((upd.abs() < half_ulp).sum())
        total += p.numel()
    assert abs(got["bf16_rounded_frac"] - rounded / total) < 1e-9, (got, rounded / total)

    assert adam_bf16_rounded(opt, names, 1e-12)["bf16_rounded_frac"] == 1.0
    # Not exactly 0: a few elements' gradients cancelled over the 3 steps
    # (m ~ 1e-10), and those updates vanish at any lr.
    assert adam_bf16_rounded(opt, names, 10.0)["bf16_rounded_frac"] < 0.01
    # Empirical: elements the bf16 step left unchanged. AdamW's own bf16
    # arithmetic differs from the fp32 formula near the threshold, so this
    # is a closeness check, not equality.
    unchanged = sum(int((before[n] == params[n].detach()).sum()) for n in names)
    assert abs(unchanged / total - got["bf16_rounded_frac"]) < 0.1, (unchanged / total, got["bf16_rounded_frac"])


if __name__ == "__main__":
    test_matches_brute_force_across_chunkings()
    test_chunk_size_is_invisible()
    test_empty_intersection_counts_ratio_not_advantage()
    test_matches_oracle_kl_definition_on_one_trajectory()
    test_bf16_rounded_fraction()
    print("tests/test_train_diagnostics.py passed")
