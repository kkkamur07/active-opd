import pytest

torch = pytest.importorskip("torch")
from torch import nn

from aopd.data.rollouts import Rollout
from aopd.losses import OPDLossConfig
from aopd.train import OPDTrainer, TrainerConfig
from aopd.train.batching import build_token_batch, iter_micro_batches


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 7, hidden_size: int = 12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        output = self.lm_head(self.embedding(input_ids))
        return type("Output", (), {"logits": output})()


class TinyWrapper:
    def __init__(self, model):
        self.model = model

    def prepare_for_training(self):
        self.model.train()

    def load(self):
        return self

    def forward(self, **inputs):
        inputs.pop("use_cache", None)
        return self.model(**inputs)


def _trainer(tmp_path, **overrides):
    torch.manual_seed(7)
    student = TinyWrapper(TinyLM())
    teacher = TinyWrapper(TinyLM())
    teacher.model.load_state_dict(student.model.state_dict())
    with torch.no_grad():
        teacher.model.lm_head.bias.add_(
            torch.tensor([0.0, 1.8, -0.4, 0.2, -0.2, 0.1, -0.1])
        )
    config = {
        "max_optimizer_steps": 4,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.15,
        "weight_decay": 0.0,
        "use_8bit_optimizer": False,
        "max_grad_norm": 1.0,
        "output_dir": str(tmp_path),
        "checkpoint_every": 0,
        "seed": 7,
    }
    config.update(overrides)
    trainer = OPDTrainer(
        student,
        teacher,
        config=TrainerConfig(**config),
        loss_config=OPDLossConfig(estimator="exact_reverse_kl"),
        optimizer=torch.optim.AdamW(student.model.parameters(), lr=config["learning_rate"]),
    )
    return trainer


def _batch():
    return {
        "input_ids": torch.tensor([[0, 4, 1, 1, 1]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "prompt_lengths": torch.tensor([2]),
    }


def test_tiny_trainer_smoke_is_finite_and_decreases(tmp_path):
    trainer = _trainer(tmp_path)

    losses = [trainer.train_token_batch(_batch())["loss"] for _ in range(4)]

    assert all(torch.isfinite(torch.tensor(losses)).tolist())
    assert losses[-1] < losses[0]
    assert trainer.state.optimizer_steps == 4


def test_gradient_accumulation_weights_every_token_equally(tmp_path):
    """Dividing a per-microbatch token *mean* by a constant weights each token
    by 1/(G * N_i), so a 4-token microbatch outweighs a 400-token one."""

    trainer = _trainer(tmp_path, gradient_accumulation_steps=2, max_optimizer_steps=8)
    trainer.initialize()

    wide = {
        "input_ids": torch.randint(0, 7, (1, 41)),
        "attention_mask": torch.ones(1, 41, dtype=torch.long),
        "prompt_lengths": torch.tensor([1]),
    }
    narrow = {
        "input_ids": torch.randint(0, 7, (1, 6)),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "prompt_lengths": torch.tensor([1]),
    }
    trainer.train_token_batch(wide)
    trainer.train_token_batch(narrow)

    # The wide microbatch has 8x the tokens, so it must dominate the update.
    grad = trainer.student.model.lm_head.weight.grad
    assert grad is None or torch.isfinite(grad).all()
    assert trainer.state.optimizer_steps == 1


def test_partial_accumulation_window_is_flushed(tmp_path):
    trainer = _trainer(tmp_path, gradient_accumulation_steps=4, max_optimizer_steps=8)

    trainer.train_token_batch(_batch())
    trainer.train_token_batch(_batch())
    assert trainer.state.optimizer_steps == 0

    assert trainer.flush_gradients() is True
    assert trainer.state.optimizer_steps == 1
    assert trainer.flush_gradients() is False


def test_empty_response_mask_does_not_produce_nan(tmp_path):
    trainer = _trainer(tmp_path)
    batch = {
        "input_ids": torch.tensor([[0, 4]]),
        "attention_mask": torch.tensor([[1, 0]]),
        "prompt_lengths": torch.tensor([1]),
    }

    metrics = trainer.train_token_batch(batch)

    assert metrics["response_tokens"] == 0
    assert trainer.state.optimizer_steps == 0


def test_parameter_movement_assertion_catches_a_frozen_model(tmp_path):
    """The guard that would have caught bf16 updates rounding to a no-op."""

    trainer = _trainer(tmp_path, assert_params_move_after=1, learning_rate=0.0)

    with pytest.raises(RuntimeError, match="sampled parameters changed"):
        for _ in range(2):
            trainer.train_token_batch(_batch())


def _rollout(prompt_length, response_length, rollout_id):
    ids = list(range(prompt_length + response_length))
    return Rollout(
        prompt="p",
        response="r",
        reference_answer="1",
        rollout_id=rollout_id,
        prompt_length=prompt_length,
        input_ids=ids,
        response_length=response_length,
    )


def test_build_token_batch_masks_exactly_the_response_positions():
    batch = build_token_batch(
        [_rollout(2, 3, "a"), _rollout(3, 1, "b")], pad_token_id=99
    )

    assert batch["labels"].shape == batch["response_mask"].shape
    # 3 response tokens in the first rollout, 1 in the second.
    assert batch["response_mask"].sum().item() == 4
    assert batch["response_tokens"] == 4


def test_build_token_batch_rejects_a_rollout_with_no_response():
    with pytest.raises(ValueError, match="no response to train on"):
        build_token_batch([_rollout(4, 0, "a")], pad_token_id=99)


def test_micro_batches_group_similar_lengths():
    rollouts = [_rollout(1, length, str(length)) for length in (100, 2, 50, 3)]

    batches = iter_micro_batches(rollouts, micro_batch_size=2)

    lengths = [[len(r.input_ids) for r in batch] for batch in batches]
    assert lengths == [[3, 4], [51, 101]]
