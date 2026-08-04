"""Response-masked on-policy distillation losses.

The training objective is the per-token reverse KL on student-visited states,

    L = E_{y ~ pi_theta} sum_t KL( pi_theta(.|x, y_<t) || pi_T(.|x, y_<t) )

computed **exactly** over the vocabulary at each visited position:

    kl_t = sum_v pi_theta(v) ( log pi_theta(v) - log pi_T(v) )

Why exactly, and not from the sampled token
-------------------------------------------
A sampled-token estimator such as veRL's ``k3`` is an unbiased estimator of the
KL *value*, but differentiating it pathwise does not give the gradient of that
value.  With ``r = log pi_theta(y) - log pi_T(y)`` and the teacher detached,
``d k3/d log pi_theta(y) = 1 - exp(-r)``, so

    E_{y~p}[ (1 - q/p) grad log p ] = sum_v (p_v - q_v) grad log p_v
                                    = grad_theta KL(q || p)

which is the *forward* KL -- the opposite direction to the one this project is
defined around.  The missing score-function term is exactly that discrepancy.
Since the trainer holds both full logit tensors anyway, the exact reverse KL is
available and is used instead; ``k1``/``k2``/``k3`` remain available as
**no-gradient diagnostics** for logging and for comparison against veRL.

Memory
------
At a real thinking-trace length the naive form is not affordable: a single
18k-token sequence costs ~5.1 GiB of bf16 logits per model and ~10.2 GiB per
fp32 ``log_softmax``, i.e. >30 GiB before weights or activations.  The loss is
therefore computed in chunks along the time axis, so peak memory scales with
``chunk_size * vocab`` rather than ``seq_len * vocab``.  Autograd recomputes
each chunk's softmax in the backward pass via checkpointing.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal

EstimatorName = Literal["exact_reverse_kl", "policy_gradient", "k1", "k2", "k3", "topk", "forward_kl"]

#: Estimators whose gradient is a consistent estimator of the reverse KL.
TRAINABLE_ESTIMATORS: frozenset[str] = frozenset(
    {"exact_reverse_kl", "policy_gradient", "k2", "topk", "forward_kl"}
)

#: Sampled estimators kept for logging/diagnostics only. Their pathwise
#: gradient does not descend the reverse KL (see the module docstring).
DIAGNOSTIC_ESTIMATORS: frozenset[str] = frozenset({"k1", "k3"})

_ESTIMATOR_ALIASES: dict[str, str] = {
    "exact": "exact_reverse_kl",
    "exact_reverse_kl": "exact_reverse_kl",
    "full_vocab": "exact_reverse_kl",
    "full_vocabulary": "exact_reverse_kl",
    "reverse_kl": "exact_reverse_kl",
    "policy_gradient": "policy_gradient",
    "pg": "policy_gradient",
    "reinforce": "policy_gradient",
    "k1": "k1",
    "k2": "k2",
    "k3": "k3",
    "topk": "topk",
    "top_k": "topk",
    "forward_kl": "forward_kl",
}


@dataclass(frozen=True)
class OPDLossConfig:
    """Hydra-friendly loss controls."""

    estimator: EstimatorName | str = "exact_reverse_kl"
    direction: str = "student_to_teacher"
    chunk_size: int = 1024
    clamp_log_ratio: float | None = 10.0
    top_k: int = 32
    reduction: Literal["mean", "sum"] = "mean"

    @classmethod
    def from_mapping(cls, config: Any) -> OPDLossConfig:
        values = dict(config)
        names = {field.name for field in fields(cls)}
        if "name" in values and "estimator" not in values:
            values["estimator"] = values["name"]
        unknown = set(values) - names - {"name"}
        if unknown:
            raise ValueError(
                f"Unknown OPD loss option(s): {sorted(unknown)}. "
                f"Known options: {sorted(names)}."
            )
        return cls(**{name: values[name] for name in names if name in values})

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        estimator = resolve_estimator(self.estimator)
        if estimator in DIAGNOSTIC_ESTIMATORS:
            raise ValueError(
                f"Estimator {estimator!r} is a diagnostic only: its pathwise gradient "
                "descends the forward KL, not the reverse KL. Use "
                "'exact_reverse_kl' (default), 'policy_gradient', or 'k2' for training, "
                "and read k1/k3 from the logged diagnostics instead."
            )
        if estimator in {"exact_reverse_kl", "policy_gradient", "k2"} and self.direction not in {
            "student_to_teacher",
            "reverse_kl",
        }:
            raise ValueError(
                f"Estimator {estimator!r} computes KL(student || teacher); "
                f"direction={self.direction!r} contradicts it."
            )
        if estimator == "forward_kl" and self.direction not in {
            "teacher_to_student",
            "forward_kl",
        }:
            raise ValueError(
                "The forward_kl estimator computes KL(teacher || student); set "
                "direction='teacher_to_student' to select it deliberately."
            )
        if self.clamp_log_ratio is not None and self.clamp_log_ratio <= 0:
            raise ValueError("clamp_log_ratio must be positive or None.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'.")


def resolve_estimator(name: Any) -> str:
    """Map a config value to a known estimator, raising on anything else.

    The previous implementation was a substring matcher ending in an
    unconditional ``return "k3"``, so a typo (or a deliberate
    ``exact_reverse_kl``) silently ran k3 while the run record logged the name
    that was asked for.
    """

    normalized = str(name).strip().lower().replace("-", "_")
    try:
        return _ESTIMATOR_ALIASES[normalized]
    except KeyError:
        raise ValueError(
            f"Unknown OPD estimator {name!r}. Known estimators: "
            f"{sorted(set(_ESTIMATOR_ALIASES.values()))}."
        ) from None


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise ImportError("OPD loss computation requires PyTorch.") from exc
    return torch


def response_token_mask(
    attention_mask: Any,
    response_start: Any,
    *,
    input_ids: Any | None = None,
    include_eos: bool = True,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> Any:
    """Build a boolean mask for response labels aligned with model logits.

    ``response_start`` is an **absolute index** into the label sequence: the
    position of the first response token.  It is *not* a token count.  The
    distinction matters under left padding, where HF requires batched
    decoder-only generation to pad on the left and a prompt-token count is no
    longer the same as the prompt's end offset.  Use
    :func:`response_start_from_lengths` to convert a prompt length plus a
    padded attention mask into the right index.

    For the usual shifted causal-LM batch (``logits[:, :-1]`` against
    ``labels = input_ids[:, 1:]``) pass the response start minus one.
    """

    torch = _require_torch()
    attention = torch.as_tensor(attention_mask, dtype=torch.bool)
    if attention.ndim == 1:
        attention = attention.unsqueeze(0)
    starts = torch.as_tensor(response_start, device=attention.device)
    if starts.ndim == 0:
        starts = starts.repeat(attention.shape[0])
    starts = starts.to(device=attention.device, dtype=torch.long).reshape(-1)
    if starts.numel() != attention.shape[0]:
        raise ValueError("response_start must contain one value per batch item.")
    positions = torch.arange(attention.shape[1], device=attention.device)
    mask = attention & (positions.unsqueeze(0) >= starts.unsqueeze(1))
    if input_ids is not None:
        labels = torch.as_tensor(input_ids, device=attention.device)
        if labels.shape != attention.shape:
            raise ValueError("input_ids and attention_mask must have the same shape.")
        if not include_eos and eos_token_id is not None:
            mask &= labels != eos_token_id
        if pad_token_id is not None and pad_token_id != eos_token_id:
            # When pad and eos share an id (common when a tokenizer has no pad
            # token) masking the id would delete the real EOS, which is the
            # token the student most needs to learn. Rely on attention instead.
            mask &= labels != pad_token_id
    return mask


def response_start_from_lengths(attention_mask: Any, prompt_lengths: Any) -> Any:
    """Convert prompt token *counts* into absolute response start indices.

    Handles left, right and no padding by locating each row's first attended
    position and offsetting by the prompt length.
    """

    torch = _require_torch()
    attention = torch.as_tensor(attention_mask)
    if attention.ndim == 1:
        attention = attention.unsqueeze(0)
    lengths = torch.as_tensor(prompt_lengths, device=attention.device)
    if lengths.ndim == 0:
        lengths = lengths.repeat(attention.shape[0])
    lengths = lengths.to(device=attention.device, dtype=torch.long).reshape(-1)
    first_attended = attention.to(torch.long).argmax(dim=1)
    return first_attended + lengths


def build_response_mask(*args: Any, **kwargs: Any) -> Any:
    """Backward-compatible descriptive alias for ``response_token_mask``."""

    return response_token_mask(*args, **kwargs)


def _clamp_log_ratio(
    log_ratio: Any,
    *,
    clamp_log_ratio: float | None = None,
    log_ratio_min: float | None = None,
    log_ratio_max: float | None = None,
) -> Any:
    torch = _require_torch()
    if clamp_log_ratio is not None:
        if clamp_log_ratio <= 0:
            raise ValueError("clamp_log_ratio must be positive or None.")
        log_ratio_min = -clamp_log_ratio
        log_ratio_max = clamp_log_ratio
    if log_ratio_min is None and log_ratio_max is None:
        return log_ratio
    minimum = log_ratio_min if log_ratio_min is not None else -float("inf")
    maximum = log_ratio_max if log_ratio_max is not None else float("inf")
    if minimum > maximum:
        raise ValueError("log_ratio_min cannot exceed log_ratio_max.")
    return torch.clamp(log_ratio, min=minimum, max=maximum)


def reverse_kl_estimator(
    student_logprobs: Any,
    teacher_logprobs: Any,
    *,
    estimator: EstimatorName | str = "k3",
    clamp_log_ratio: float | None = None,
    log_ratio_min: float | None = None,
    log_ratio_max: float | None = None,
) -> Any:
    """Sampled per-token estimators of the reverse KL **value**.

    ``k1`` is unbiased for the value but has zero expected gradient; ``k3`` is
    non-negative and low variance but its pathwise gradient descends the
    forward KL.  Use these for logging only -- :func:`compute_opd_loss` refuses
    to train on them.
    """

    torch = _require_torch()
    if student_logprobs.shape != teacher_logprobs.shape:
        raise ValueError("Student and teacher log-probabilities must have equal shape.")
    name = resolve_estimator(estimator)
    if name not in {"k1", "k2", "k3"}:
        raise ValueError(
            f"Sampled reverse-KL estimator must be k1, k2, or k3; got {estimator!r}."
        )
    log_ratio = _clamp_log_ratio(
        student_logprobs - teacher_logprobs,
        clamp_log_ratio=clamp_log_ratio,
        log_ratio_min=log_ratio_min,
        log_ratio_max=log_ratio_max,
    )
    if name == "k1":
        return log_ratio
    if name == "k2":
        return 0.5 * log_ratio.square()
    return torch.exp(-log_ratio) - 1.0 + log_ratio


def _exact_reverse_kl_chunk(student_logits: Any, teacher_logits: Any) -> Any:
    """Exact per-token ``KL(student || teacher)`` over the full vocabulary."""

    torch = _require_torch()
    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    with torch.no_grad():
        teacher_log_probs = torch.log_softmax(teacher_logits.float(), dim=-1)
    student_probs = student_log_probs.exp()
    return (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)


def _forward_kl_chunk(student_logits: Any, teacher_logits: Any) -> Any:
    """Exact per-token ``KL(teacher || student)`` over the full vocabulary."""

    torch = _require_torch()
    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    with torch.no_grad():
        teacher_log_probs = torch.log_softmax(teacher_logits.float(), dim=-1)
        teacher_probs = teacher_log_probs.exp()
    return (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)


def _topk_reverse_kl_chunk(student_logits: Any, teacher_logits: Any, top_k: int) -> Any:
    """Reverse KL restricted to the union of both models' top-k, plus a tail.

    The previous implementation renormalized over the teacher's top-k only,
    which made the objective blind to all student mass outside that set (its
    gradient there was identically zero).  Here the residual probability of
    each model is kept as one lumped bucket so off-support mass still
    contributes.
    """

    torch = _require_torch()
    vocabulary_size = student_logits.shape[-1]
    top_k = min(top_k, vocabulary_size)
    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    with torch.no_grad():
        teacher_log_probs = torch.log_softmax(teacher_logits.float(), dim=-1)
        teacher_ids = torch.topk(teacher_log_probs, k=top_k, dim=-1).indices
        student_ids = torch.topk(student_log_probs.detach(), k=top_k, dim=-1).indices
        candidate_ids = torch.cat([teacher_ids, student_ids], dim=-1)

    student_subset = student_log_probs.gather(-1, candidate_ids)
    teacher_subset = teacher_log_probs.gather(-1, candidate_ids)
    # De-duplicate the union: a repeated id keeps its first occurrence only.
    first = torch.zeros_like(candidate_ids, dtype=torch.bool)
    sorted_ids, order = candidate_ids.sort(dim=-1)
    unique = torch.ones_like(sorted_ids, dtype=torch.bool)
    unique[..., 1:] = sorted_ids[..., 1:] != sorted_ids[..., :-1]
    first.scatter_(-1, order, unique)

    student_probs = student_subset.exp() * first
    teacher_probs = teacher_subset.exp() * first
    selected = (student_probs * (student_subset - teacher_subset)).sum(dim=-1)
    student_tail = (1.0 - student_probs.sum(dim=-1)).clamp_min(1e-9)
    teacher_tail = (1.0 - teacher_probs.sum(dim=-1)).clamp_min(1e-9)
    return selected + student_tail * (student_tail.log() - teacher_tail.log())


_CHUNK_FUNCTIONS = {
    "exact_reverse_kl": _exact_reverse_kl_chunk,
    "forward_kl": _forward_kl_chunk,
}


def per_token_divergence(
    student_logits: Any,
    teacher_logits: Any,
    *,
    estimator: str = "exact_reverse_kl",
    chunk_size: int = 1024,
    top_k: int = 32,
) -> Any:
    """Per-token divergence over ``[batch, time]``, chunked along time.

    Peak activation memory scales with ``chunk_size * vocab`` instead of
    ``time * vocab``; each chunk's softmax is recomputed during backward via
    gradient checkpointing, so an 18k-token sequence no longer needs tens of
    GiB of resident log-probabilities.
    """

    torch = _require_torch()
    from torch.utils.checkpoint import checkpoint

    name = resolve_estimator(estimator)
    if name == "topk":
        def chunk_fn(student_chunk: Any, teacher_chunk: Any) -> Any:
            return _topk_reverse_kl_chunk(student_chunk, teacher_chunk, top_k)
    else:
        try:
            chunk_fn = _CHUNK_FUNCTIONS[name]
        except KeyError:
            raise ValueError(
                f"Estimator {name!r} has no full-vocabulary chunk implementation."
            ) from None

    time_steps = student_logits.shape[1]
    if chunk_size >= time_steps:
        return chunk_fn(student_logits, teacher_logits)

    outputs = []
    requires_grad = student_logits.requires_grad and torch.is_grad_enabled()
    for start in range(0, time_steps, chunk_size):
        stop = min(start + chunk_size, time_steps)
        student_chunk = student_logits[:, start:stop]
        teacher_chunk = teacher_logits[:, start:stop]
        if requires_grad:
            outputs.append(
                checkpoint(chunk_fn, student_chunk, teacher_chunk, use_reentrant=False)
            )
        else:
            outputs.append(chunk_fn(student_chunk, teacher_chunk))
    return torch.cat(outputs, dim=1)


def masked_mean(
    values: Any,
    mask: Any,
    *,
    weights: Any | None = None,
    reduction: str = "mean",
) -> Any:
    """Reduce token values without producing NaNs for an empty response.

    ``weights`` is an optional per-sequence weight, broadcast over time.  It is
    the hook a future acquisition score plugs into: a hard filter is the
    special case ``weights in {0, 1}``.
    """

    torch = _require_torch()
    if mask.dtype != torch.bool:
        raise TypeError(
            "response_mask must be boolean; a float mask silently degrades to "
            "its nonzero pattern and cannot express a weight. Pass per-sequence "
            "weights via the 'weights' argument instead."
        )
    weight = mask.to(dtype=values.dtype)
    if weights is not None:
        sequence_weights = torch.as_tensor(weights, device=values.device, dtype=values.dtype)
        weight = weight * sequence_weights.reshape(-1, *([1] * (weight.ndim - 1)))
    # torch.where avoids inf * 0 -> NaN at masked positions.
    safe_values = torch.where(mask, values, torch.zeros_like(values))
    total = (safe_values * weight).sum()
    if reduction == "sum":
        return total
    if reduction != "mean":
        raise ValueError("reduction must be 'mean' or 'sum'.")
    denominator = weight.sum().clamp_min(1e-6)
    return total / denominator


def opd_loss_terms(
    student_logits: Any,
    teacher_logits: Any,
    labels: Any,
    response_mask: Any,
    config: OPDLossConfig | dict[str, Any] | None = None,
    *,
    weights: Any | None = None,
) -> tuple[Any, Any]:
    """Return ``(weighted_sum, weight_total)`` for one micro-batch.

    Returning the unreduced pair is what makes gradient accumulation correct:
    the caller sums both across the accumulation window and divides once, so
    every token carries equal weight regardless of how micro-batches were cut.
    Dividing a per-micro-batch mean by a constant does not do that.
    """

    torch = _require_torch()
    resolved = (
        config
        if isinstance(config, OPDLossConfig)
        else OPDLossConfig.from_mapping(config or {})
    )
    resolved.validate()
    estimator = resolve_estimator(resolved.estimator)

    mask = response_mask
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    mask = mask.to(device=student_logits.device)
    if mask.dtype != torch.bool:
        raise TypeError("response_mask must be a boolean tensor.")
    if mask.shape != student_logits.shape[:-1]:
        raise ValueError("response_mask must match the first dimensions of logits.")

    teacher_logits = teacher_logits.detach()

    if estimator == "policy_gradient":
        student_selected, teacher_selected = _sampled_logprobs(
            student_logits, teacher_logits, labels
        )
        log_ratio = student_selected - teacher_selected.detach()
        reward = _clamp_log_ratio(
            log_ratio.detach(), clamp_log_ratio=resolved.clamp_log_ratio
        )
        # d/dtheta E_p[log p - log q] = E_p[(log p/q + 1) grad log p]; the +1
        # term vanishes in expectation since E_p[grad log p] = 0.
        values = student_selected * (reward + 1.0)
    elif estimator == "k2":
        student_selected, teacher_selected = _sampled_logprobs(
            student_logits, teacher_logits, labels
        )
        log_ratio = _clamp_log_ratio(
            student_selected - teacher_selected.detach(),
            clamp_log_ratio=resolved.clamp_log_ratio,
        )
        values = 0.5 * log_ratio.square()
    else:
        values = per_token_divergence(
            student_logits,
            teacher_logits,
            estimator=estimator,
            chunk_size=resolved.chunk_size,
            top_k=resolved.top_k,
        )

    weight = mask.to(dtype=values.dtype)
    if weights is not None:
        sequence_weights = torch.as_tensor(
            weights, device=values.device, dtype=values.dtype
        )
        if sequence_weights.numel() != mask.shape[0]:
            raise ValueError("weights must contain one value per batch item.")
        weight = weight * sequence_weights.reshape(-1, *([1] * (weight.ndim - 1)))
    safe_values = torch.where(mask, values, torch.zeros_like(values))
    return (safe_values * weight).sum(), weight.sum()


def _sampled_logprobs(
    student_logits: Any,
    teacher_logits: Any,
    labels: Any,
) -> tuple[Any, Any]:
    torch = _require_torch()
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have equal shape.")
    if labels is None:
        raise ValueError("Sampled estimators require labels.")
    labels = torch.as_tensor(labels, device=student_logits.device)
    if labels.shape != student_logits.shape[:-1]:
        raise ValueError("labels must match the first dimensions of logits.")
    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    with torch.no_grad():
        teacher_log_probs = torch.log_softmax(teacher_logits.float(), dim=-1)
    token_labels = labels.to(dtype=torch.long)
    student_selected = student_log_probs.gather(-1, token_labels.unsqueeze(-1)).squeeze(-1)
    teacher_selected = teacher_log_probs.gather(-1, token_labels.unsqueeze(-1)).squeeze(-1)
    return student_selected, teacher_selected


def compute_opd_loss(
    student_logits: Any,
    teacher_logits: Any,
    labels: Any,
    response_mask: Any,
    config: OPDLossConfig | dict[str, Any] | None = None,
    *,
    weights: Any | None = None,
) -> Any:
    """Compute a response-masked OPD loss for one tokenized batch."""

    resolved = (
        config
        if isinstance(config, OPDLossConfig)
        else OPDLossConfig.from_mapping(config or {})
    )
    total, weight = opd_loss_terms(
        student_logits,
        teacher_logits,
        labels,
        response_mask,
        resolved,
        weights=weights,
    )
    if resolved.reduction == "sum":
        return total
    return total / weight.clamp_min(1e-6)


def sampled_diagnostics(
    student_logits: Any,
    teacher_logits: Any,
    labels: Any,
    response_mask: Any,
    *,
    clamp_log_ratio: float | None = 10.0,
) -> dict[str, float]:
    """Log-only sampled KL estimators, computed without building a graph."""

    torch = _require_torch()
    with torch.no_grad():
        student_selected, teacher_selected = _sampled_logprobs(
            student_logits, teacher_logits, labels
        )
        mask = response_mask.to(device=student_selected.device, dtype=torch.bool)
        denominator = mask.sum().clamp_min(1)
        out: dict[str, float] = {}
        for name in ("k1", "k2", "k3"):
            values = reverse_kl_estimator(
                student_selected,
                teacher_selected,
                estimator=name,
                clamp_log_ratio=clamp_log_ratio,
            )
            out[f"kl_{name}"] = float(
                (torch.where(mask, values, torch.zeros_like(values)).sum() / denominator).item()
            )
        log_ratio = student_selected - teacher_selected
        out["log_ratio_mean"] = float(
            (torch.where(mask, log_ratio, torch.zeros_like(log_ratio)).sum() / denominator).item()
        )
        if clamp_log_ratio is not None:
            clipped = (log_ratio.abs() > clamp_log_ratio) & mask
            out["clamped_fraction"] = float((clipped.sum() / denominator).item())
    return out


def compute_policy_gradient_opd_loss(
    student_logprobs: Any,
    teacher_logprobs: Any,
    response_mask: Any,
    *,
    clamp_log_ratio: float | None = None,
) -> Any:
    """Score-function estimator of the reverse-KL gradient on sampled tokens.

    ``E_p[(log p/q + 1) grad log p] = grad KL(p||q)``.  Kept as a public API
    for the sampled path; the trainer's default is the exact chunked form.
    """

    torch = _require_torch()
    reward = (student_logprobs - teacher_logprobs.detach()).detach()
    if clamp_log_ratio is not None:
        reward = _clamp_log_ratio(reward, clamp_log_ratio=clamp_log_ratio)
    values = student_logprobs * (reward + 1.0)
    mask = response_mask.to(device=values.device, dtype=torch.bool)
    total = torch.where(mask, values, torch.zeros_like(values)).sum()
    return total / mask.sum().clamp_min(1)


opd_loss = compute_opd_loss
compute_kl = reverse_kl_estimator


__all__ = [
    "DIAGNOSTIC_ESTIMATORS",
    "TRAINABLE_ESTIMATORS",
    "EstimatorName",
    "OPDLossConfig",
    "build_response_mask",
    "compute_kl",
    "compute_opd_loss",
    "compute_policy_gradient_opd_loss",
    "masked_mean",
    "opd_loss",
    "opd_loss_terms",
    "per_token_divergence",
    "resolve_estimator",
    "response_start_from_lengths",
    "response_token_mask",
    "reverse_kl_estimator",
    "sampled_diagnostics",
]
