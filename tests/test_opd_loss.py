import pytest

torch = pytest.importorskip("torch")

from aopd.losses.opd import (
    OPDLossConfig,
    compute_opd_loss,
    per_token_divergence,
    resolve_estimator,
    reverse_kl_estimator,
)


def _analytic_reverse_kl_grad(theta, teacher):
    log_p = torch.log_softmax(theta, -1)
    log_q = torch.log_softmax(teacher, -1)
    (grad,) = torch.autograd.grad((log_p.exp() * (log_p - log_q)).sum(), theta)
    return grad


@pytest.mark.parametrize("estimator", ["exact_reverse_kl", "policy_gradient", "k2"])
def test_expected_gradient_equals_analytic_reverse_kl(estimator):
    """The property the whole project rests on, and the one nothing tested.

    The old default (k3 differentiated pathwise) passes every value-based
    assertion while its expected gradient is that of the *forward* KL.
    """

    torch.manual_seed(0)
    vocab = 5
    theta = torch.randn(vocab, requires_grad=True)
    teacher = torch.randn(vocab)
    want = _analytic_reverse_kl_grad(theta, teacher)

    probs = torch.log_softmax(theta, -1).exp().detach()
    got = torch.zeros(vocab)
    for token in range(vocab):  # exact expectation over y ~ student
        candidate = theta.detach().clone().requires_grad_(True)
        loss = compute_opd_loss(
            candidate.view(1, 1, vocab),
            teacher.view(1, 1, vocab),
            torch.tensor([[token]]),
            torch.ones(1, 1, dtype=torch.bool),
            OPDLossConfig(estimator=estimator, clamp_log_ratio=None),
        )
        got = got + probs[token] * torch.autograd.grad(loss, candidate)[0]

    assert torch.allclose(got, want, atol=1e-5), f"{estimator}: {got} != {want}"


@pytest.mark.parametrize("estimator", ["k1", "k3"])
def test_sampled_diagnostic_estimators_are_refused_for_training(estimator):
    with pytest.raises(ValueError, match="diagnostic only"):
        OPDLossConfig(estimator=estimator)


def test_unknown_estimator_names_raise_instead_of_silently_becoming_k3():
    """`_estimator_from_name` used to end in an unconditional `return "k3"`.

    Writing `estimator: exact_reverse_kl` therefore ran k3 and logged the name
    that was asked for, so the estimator ablation compared k3 to itself.
    """

    for name in ("exact_reverse_kl_typo", "k5", "analytic", ""):
        with pytest.raises(ValueError, match="Unknown OPD estimator"):
            resolve_estimator(name)


def test_unknown_config_keys_raise():
    with pytest.raises(ValueError, match="Unknown OPD loss option"):
        OPDLossConfig.from_mapping({"estimator": "k2", "clamp": 5.0})


def test_chunking_is_numerically_identical_to_the_unchunked_path():
    torch.manual_seed(1)
    student = torch.randn(2, 96, 32, requires_grad=True)
    teacher = torch.randn(2, 96, 32)

    whole = per_token_divergence(student, teacher, chunk_size=10_000)
    chunked = per_token_divergence(student, teacher, chunk_size=7)

    assert torch.allclose(whole, chunked, atol=1e-6)
    (grad_whole,) = torch.autograd.grad(whole.sum(), student, retain_graph=True)
    (grad_chunked,) = torch.autograd.grad(chunked.sum(), student)
    assert torch.allclose(grad_whole, grad_chunked, atol=1e-6)


def test_response_mask_controls_loss_and_gradient():
    student_logits = torch.randn(1, 3, 5, requires_grad=True)
    teacher_logits = torch.randn(1, 3, 5, requires_grad=True)
    labels = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[True, False, False]])

    loss = compute_opd_loss(
        student_logits,
        teacher_logits,
        labels,
        mask,
        OPDLossConfig(estimator="exact_reverse_kl"),
    )
    loss.backward()

    assert loss.ndim == 0
    assert student_logits.grad is not None
    assert torch.all(student_logits.grad[:, 1:] == 0)
    assert teacher_logits.grad is None


def test_float_mask_is_rejected_so_weights_cannot_be_smuggled_in():
    """A float mask used to be cast to bool, silently discarding the weight."""

    student_logits = torch.randn(1, 2, 4, requires_grad=True)
    teacher_logits = torch.randn(1, 2, 4)

    with pytest.raises(TypeError, match="boolean"):
        compute_opd_loss(
            student_logits,
            teacher_logits,
            torch.tensor([[1, 2]]),
            torch.full((1, 2), 0.25),
            OPDLossConfig(estimator="exact_reverse_kl"),
        )


def test_per_sequence_weights_scale_the_loss():
    """A hard filter is the weight interface with weights in {0, 1}."""

    torch.manual_seed(2)
    student_logits = torch.randn(2, 2, 4, requires_grad=True)
    teacher_logits = torch.randn(2, 2, 4)
    labels = torch.tensor([[1, 2], [0, 3]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    config = OPDLossConfig(estimator="exact_reverse_kl")

    both = compute_opd_loss(student_logits, teacher_logits, labels, mask, config)
    first_only = compute_opd_loss(
        student_logits, teacher_logits, labels, mask, config, weights=[1.0, 0.0]
    )

    assert not torch.isclose(both, first_only)


def test_topk_keeps_gradient_on_student_mass_outside_the_teacher_top_k():
    """The old top-k renormalized over the teacher's support only, so all
    student mass outside it had exactly zero gradient."""

    torch.manual_seed(3)
    student = torch.randn(1, 1, 20, requires_grad=True)
    teacher = torch.randn(1, 1, 20)
    loss = compute_opd_loss(
        student,
        teacher,
        torch.tensor([[0]]),
        torch.ones(1, 1, dtype=torch.bool),
        OPDLossConfig(estimator="topk", top_k=3),
    )
    (grad,) = torch.autograd.grad(loss, student)

    teacher_top = torch.topk(teacher[0, 0], 3).indices.tolist()
    outside = [index for index in range(20) if index not in teacher_top]
    assert grad[0, 0, outside].abs().max() > 0


def test_k3_remains_available_as_a_value_diagnostic():
    matching = torch.zeros(4)
    different = torch.tensor([-2.0, -0.5, 0.5, 2.0])

    assert torch.allclose(
        reverse_kl_estimator(matching, matching, estimator="k3"), torch.zeros(4)
    )
    assert torch.all(
        reverse_kl_estimator(different, torch.zeros(4), estimator="k3") >= 0
    )
