"""Split rollout/training orchestration for the local Active OPD prototype."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aopd.data.rollouts import ALL_OUTCOMES, Rollout
from aopd.losses.opd import (
    OPDLossConfig,
    opd_loss_terms,
    response_start_from_lengths,
    response_token_mask,
)
from aopd.utils.logging import JsonlMetricsLogger
from aopd.utils.reproducibility import peak_cuda_memory, seed_everything

from .batching import build_token_batch, iter_micro_batches
from .rollouts import RolloutCollector
from .selector import SelectionResult, VerifiedWrongSelector


@dataclass(frozen=True)
class TrainerConfig:
    """Memory-conscious optimizer and checkpoint controls."""

    #: Optimizer steps to run. Distinct from ``max_rounds``: a round is one
    #: generate/verify/select cycle and usually contains many optimizer steps.
    max_optimizer_steps: int = 1000
    max_rounds: int | None = None
    gradient_accumulation_steps: int = 16
    micro_batch_size: int = 1
    #: Cap on ``batch * padded_length`` per micro-batch. With 18k-token traces
    #: a fixed micro_batch_size is not enough to bound activation memory.
    max_micro_batch_tokens: int | None = None
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-8
    use_8bit_optimizer: bool = False
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    output_dir: str = "outputs/active-opd"
    checkpoint_every: int = 0
    seed: int = 42
    deterministic: bool = False
    amp_enabled: bool | None = None
    amp_dtype: str = "auto"
    amp_grad_scaler: bool | None = None
    #: Keep parameters in fp32 so small updates are not rounded away. bf16
    #: storage has an 8-bit mantissa: at lr=1e-5 an Adam step lands below half
    #: an ulp for most weights and ~85-90% of parameters never change. Compute
    #: still runs in bf16 under autocast.
    master_weights: str = "fp32"
    #: Fail fast if parameters are not actually moving after N optimizer steps.
    assert_params_move_after: int = 0

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> TrainerConfig:
        names = {field.name for field in cls.__dataclass_fields__.values()}
        values = {name: config[name] for name in names if name in config}
        aliases = {
            "autocast_enabled": "amp_enabled",
            "autocast_dtype": "amp_dtype",
            "grad_scaler": "amp_grad_scaler",
            "use_grad_scaler": "amp_grad_scaler",
            "compute_dtype": "amp_dtype",
            "use_8bit": "use_8bit_optimizer",
            "max_steps": "max_optimizer_steps",
        }
        for source, target in aliases.items():
            if source in config and target not in values:
                values[target] = config[source]
        if "betas" in values:
            values["betas"] = tuple(values["betas"])
        return cls(**values)


@dataclass
class TrainerState:
    """Serializable progress metadata for checkpoints and metrics."""

    update_step: int = 0
    optimizer_steps: int = 0
    rounds: int = 0
    generated_rollouts: int = 0
    retained_rollouts: int = 0
    #: Response tokens actually backpropagated. This, not rollout count, is the
    #: training budget that must be matched across baselines.
    response_tokens_trained: int = 0
    generated_tokens: int = 0
    verifier_outcomes: dict[str, int] = field(default_factory=dict)
    last_loss: float | None = None


class OPDTrainer:
    """Train a student on selected, already-collected rollouts.

    Rollout generation and answer selection remain injectable.  This keeps the
    local prototype testable and leaves a direct migration point to veRL's
    rollout workers later.
    """

    def __init__(
        self,
        student: Any,
        teacher: Any,
        *,
        config: TrainerConfig | Mapping[str, Any] | None = None,
        loss_config: OPDLossConfig | Mapping[str, Any] | None = None,
        optimizer: Any | None = None,
        logger: JsonlMetricsLogger | None = None,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.config = (
            config
            if isinstance(config, TrainerConfig)
            else TrainerConfig.from_mapping(config or {})
        )
        self.loss_config = (
            loss_config
            if isinstance(loss_config, OPDLossConfig)
            else OPDLossConfig.from_mapping(loss_config or {})
        )
        if self.config.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.config.max_optimizer_steps <= 0:
            raise ValueError("max_optimizer_steps must be positive.")
        if self.config.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive.")
        self.optimizer = optimizer
        self.logger = logger or JsonlMetricsLogger(
            Path(self.config.output_dir) / "metrics.jsonl"
        )
        self.state = TrainerState()
        self._micro_steps = 0
        self._initialized = False
        self._amp_enabled = False
        self._amp_dtype: Any | None = None
        self._amp_device_type: str | None = None
        self._grad_scaler: Any | None = None
        self._scheduler: Any | None = None
        # Running totals for the current accumulation window. Accumulating the
        # weighted sum and the token count separately is what makes every token
        # carry equal weight; dividing each micro-batch by its own token count
        # and then by a constant does not.
        self._window_loss_sum = 0.0
        self._window_tokens = 0.0
        self._param_snapshot: dict[str, Any] | None = None

    @property
    def amp_enabled(self) -> bool:
        """Whether autocast is active for the initialized trainer."""

        return self._amp_enabled

    @property
    def active_amp_dtype(self) -> str | None:
        """Name of the resolved autocast dtype, if autocast is active."""

        return self._dtype_name(self._amp_dtype)

    @property
    def grad_scaler_enabled(self) -> bool:
        """Whether FP16 gradient scaling is active."""

        return self._grad_scaler is not None

    @property
    def autocast_enabled(self) -> bool:
        """Compatibility alias for the resolved autocast state."""

        return self.amp_enabled

    @property
    def autocast_dtype(self) -> str | None:
        """Compatibility alias for the resolved autocast dtype."""

        return self.active_amp_dtype

    def initialize(self) -> OPDTrainer:
        """Load models and create the optimizer at the explicit runtime boundary."""

        seed_everything(self.config.seed, deterministic=self.config.deterministic)
        prepare_student = getattr(self.student, "prepare_for_training", None)
        if callable(prepare_student):
            prepare_student()
        else:
            load_student = getattr(self.student, "load", None)
            if callable(load_student):
                load_student()
            model = getattr(self.student, "model", None)
            if model is not None and hasattr(model, "train"):
                model.train()
        load_teacher = getattr(self.teacher, "load", None)
        if callable(load_teacher):
            load_teacher()
        self._freeze_teacher()
        master_dtype = self._apply_master_weights()
        self._configure_amp()
        if self.optimizer is None:
            self.optimizer = self._build_optimizer()
        self._build_scheduler()
        self._zero_grad()
        self._initialized = True
        if self.config.assert_params_move_after > 0:
            self._param_snapshot = self._snapshot_parameters()
        self.logger.log(
            "initialized",
            seed=self.config.seed,
            loss=asdict(self.loss_config),
            estimator=self.loss_config.estimator,
            amp_enabled=self._amp_enabled,
            amp_dtype=self._dtype_name(self._amp_dtype),
            amp_device_type=self._amp_device_type,
            amp_grad_scaler=self._grad_scaler is not None,
            master_weights=self._dtype_name(master_dtype),
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            micro_batch_size=self.config.micro_batch_size,
        )
        return self

    def _apply_master_weights(self) -> Any:
        """Keep trainable parameters in fp32 unless explicitly told not to.

        With bf16 storage an Adam step at lr=1e-5 is smaller than half an ulp
        for most weights, so ``p.add_(update)`` rounds to a no-op and the model
        barely trains while the loss still decreases. Compute stays in bf16 via
        autocast; only the master copy is fp32.
        """

        try:
            import torch
        except ImportError:  # pragma: no cover - training already needs torch
            return None
        model = getattr(self.student, "model", None)
        if model is None:
            return None
        try:
            current = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            return None
        setting = str(self.config.master_weights).lower()
        if setting in {"none", "off", "false"}:
            if current in {torch.bfloat16, torch.float16}:
                self.logger.log(
                    "master_weights_disabled",
                    dtype=self._dtype_name(current),
                    warning=(
                        "Optimizing low-precision parameters directly: small updates "
                        "will be rounded away. Use master_weights='fp32' or a "
                        "stochastic-rounding optimizer."
                    ),
                )
            return current
        if setting not in {"fp32", "float32", "auto"}:
            raise ValueError(
                f"Unsupported master_weights={self.config.master_weights!r}; "
                "use 'fp32' or 'none'."
            )
        if current == torch.float32:
            return current
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.dtype in {
                torch.bfloat16,
                torch.float16,
            }:
                parameter.data = parameter.data.to(torch.float32)
        return torch.float32

    def _build_optimizer(self) -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError("Training requires PyTorch.") from exc
        model = self.student.model
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if self.config.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
            except ImportError:
                bnb = None
            if bnb is not None:
                return bnb.optim.AdamW8bit(
                    parameters,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    betas=self.config.betas,
                    eps=self.config.eps,
                )
        return torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
            eps=self.config.eps,
        )

    def _build_scheduler(self) -> None:
        """Linear warmup, so every arm shares one LR schedule by global step."""

        if self.config.warmup_steps <= 0 or self.optimizer is None:
            return
        try:
            import torch
        except ImportError:  # pragma: no cover - training already needs torch
            return
        warmup = self.config.warmup_steps

        def lr_lambda(step: int) -> float:
            return min(1.0, (step + 1) / warmup)

        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _snapshot_parameters(self, limit: int = 6) -> dict[str, Any]:
        model = getattr(self.student, "model", None)
        if model is None:
            return {}
        snapshot: dict[str, Any] = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            snapshot[name] = parameter.detach().clone()
            if len(snapshot) >= limit:
                break
        return snapshot

    def _check_parameters_moved(self) -> None:
        """Fail loudly if the optimizer is not actually changing weights."""

        if not self._param_snapshot:
            return
        model = getattr(self.student, "model", None)
        if model is None:
            return
        current = dict(model.named_parameters())
        moved = []
        for name, before in self._param_snapshot.items():
            parameter = current.get(name)
            if parameter is None:
                continue
            moved.append(float((parameter.detach() != before).float().mean().item()))
        self._param_snapshot = None
        if not moved:
            return
        fraction = sum(moved) / len(moved)
        self.logger.log(
            "parameter_movement",
            fraction_changed=fraction,
            steps=self.state.optimizer_steps,
        )
        if fraction < 0.5:
            raise RuntimeError(
                f"Only {fraction:.1%} of sampled parameters changed after "
                f"{self.state.optimizer_steps} optimizer steps. This usually means "
                "the optimizer is writing into low-precision weights (set "
                "master_weights='fp32') or the learning rate is too small."
            )

    def train_token_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Run one microbatch on shifted causal-LM inputs."""

        if not self._initialized:
            self.initialize()
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise ValueError("batch must contain input_ids.")
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        labels = batch.get("labels")
        if labels is None:
            model_input_ids = input_ids[:, :-1]
            model_attention = attention_mask[:, :-1]
            labels = input_ids[:, 1:]
            prompt_lengths = batch.get("prompt_lengths")
            if prompt_lengths is None:
                raise ValueError(
                    "batch must contain prompt_lengths when labels are omitted."
                )
            # Absolute response start, valid for left, right and no padding.
            response_start = response_start_from_lengths(
                attention_mask, prompt_lengths
            )
            response_mask = response_token_mask(
                attention_mask[:, 1:],
                response_start - 1,
                input_ids=labels,
                include_eos=batch.get("include_eos", True),
                eos_token_id=batch.get("eos_token_id"),
                pad_token_id=batch.get("pad_token_id"),
            )
        else:
            model_input_ids = batch.get("model_input_ids", input_ids)
            model_attention = batch.get("model_attention_mask", attention_mask)
            response_mask = batch.get("response_mask")
            if response_mask is None:
                raise ValueError(
                    "batch must contain response_mask when labels are supplied."
                )

        weights = batch.get("weights")

        student_inputs = self._move_to_model_device(
            self.student,
            {
                "input_ids": model_input_ids,
                "attention_mask": model_attention,
            },
        )
        teacher_inputs = self._move_to_model_device(
            self.teacher,
            {
                "input_ids": model_input_ids,
                "attention_mask": model_attention,
            },
        )
        with self._autocast_context():
            student_outputs = self.student.forward(**student_inputs)
            with self._no_grad():
                teacher_outputs = self.teacher.forward(
                    **teacher_inputs, use_cache=False
                )
            student_logits = self._extract_logits(student_outputs)
            teacher_logits = self._extract_logits(teacher_outputs).to(
                device=student_logits.device
            )
            labels = labels.to(student_logits.device)
            response_mask = response_mask.to(student_logits.device)
            # Unreduced terms: the accumulation window divides once, at the end.
            loss_sum, token_count = opd_loss_terms(
                student_logits,
                teacher_logits,
                labels,
                response_mask,
                self.loss_config,
                weights=weights,
            )
        # Free the teacher's activations before the student's backward pass.
        del teacher_outputs, teacher_logits

        tokens = float(token_count.detach().item())
        if tokens <= 0:
            self.logger.log("train_skipped", reason="empty response mask")
            return {
                "loss": float("nan"),
                "response_tokens": 0,
                "update_step": self.state.update_step,
                "optimizer_steps": self.state.optimizer_steps,
                "optimizer_step": False,
            }

        # Normalize by the *window's* expected token count so each token's
        # contribution is independent of how micro-batches were cut. The final
        # division happens in _clip_and_step via the accumulated denominator.
        surrogate = loss_sum / max(tokens, 1.0)
        scaled = surrogate * (tokens / max(self._expected_window_tokens(tokens), 1.0))
        if self._grad_scaler is not None:
            self._grad_scaler.scale(scaled).backward()
        else:
            scaled.backward()

        self._window_loss_sum += float(loss_sum.detach().item())
        self._window_tokens += tokens
        self._micro_steps += 1
        optimizer_step = False
        if self._micro_steps % self.config.gradient_accumulation_steps == 0:
            self._clip_and_step()
            optimizer_step = True

        loss_value = float(loss_sum.detach().item()) / tokens
        self.state.update_step += 1
        self.state.optimizer_steps += int(optimizer_step)
        self.state.last_loss = loss_value
        self.state.response_tokens_trained += int(tokens)
        metrics = {
            "loss": loss_value,
            "response_tokens": int(tokens),
            "update_step": self.state.update_step,
            "optimizer_steps": self.state.optimizer_steps,
            "optimizer_step": optimizer_step,
            "response_tokens_trained": self.state.response_tokens_trained,
            "learning_rate": self._current_lr(),
            "peak_memory_bytes": peak_cuda_memory(),
        }
        self.logger.log("train", **metrics)
        if (
            self.config.assert_params_move_after > 0
            and self.state.optimizer_steps >= self.config.assert_params_move_after
        ):
            self._check_parameters_moved()
        if (
            self.config.checkpoint_every > 0
            and optimizer_step
            and self.state.optimizer_steps % self.config.checkpoint_every == 0
        ):
            self.save_checkpoint()
        return metrics

    def _expected_window_tokens(self, current_tokens: float) -> float:
        """Estimate the accumulation window's total token count.

        The exact total is unknown until the window closes, so the running mean
        is used as the scale. This keeps gradient magnitude stable across
        windows without needing a second pass over the data.
        """

        seen = self._micro_steps % self.config.gradient_accumulation_steps
        if seen == 0 or self._window_tokens <= 0:
            return current_tokens * self.config.gradient_accumulation_steps
        mean_tokens = self._window_tokens / seen
        return mean_tokens * self.config.gradient_accumulation_steps

    def _current_lr(self) -> float | None:
        if self.optimizer is None:
            return None
        groups = getattr(self.optimizer, "param_groups", None)
        if not groups:
            return None
        return float(groups[0].get("lr", 0.0))

    def flush_gradients(self) -> bool:
        """Apply a partial accumulation window, if one is open.

        Without this the last few micro-batches of a run are computed and then
        discarded.
        """

        if self._micro_steps % self.config.gradient_accumulation_steps == 0:
            return False
        self._clip_and_step()
        self.state.optimizer_steps += 1
        self._micro_steps = 0
        return True

    def fit_rollout_rounds(
        self,
        prompts: Iterable[Any],
        *,
        batch_builder: Callable[[Sequence[Rollout]], Mapping[str, Any]] | None = None,
        collector: RolloutCollector | None = None,
        selector: Any | None = None,
        generation: Any | None = None,
        references: Sequence[str | None] | None = None,
        rounds: int | None = None,
        on_round_end: Callable[[int, TrainerState], None] | None = None,
    ) -> TrainerState:
        """Run collection, verification, selection, and optimization rounds.

        Each round regenerates rollouts with the *current* weights, so the
        states stay on-policy; within a round the selection is split into
        micro-batches rather than forced into one giant batch.
        """

        if not self._initialized:
            self.initialize()
        collector = collector or RolloutCollector(self.student)
        selector = selector or VerifiedWrongSelector()
        if batch_builder is None:
            batch_builder = self._default_batch_builder()
        prompt_list = list(prompts)
        total_rounds = rounds if rounds is not None else self.config.max_rounds
        if total_rounds is None:
            total_rounds = self.config.max_optimizer_steps
        if total_rounds <= 0:
            return self.state

        for round_index in range(total_rounds):
            rollouts = collector.collect(
                prompt_list,
                generation=generation,
                references=references,
                round_index=round_index,
            )
            selection: SelectionResult = selector.select(rollouts)
            self.state.rounds += 1
            self.state.generated_rollouts += len(rollouts)
            self.state.retained_rollouts += len(selection.selected)
            self.state.generated_tokens += sum(
                int(rollout.response_length or 0) for rollout in rollouts
            )
            for key in ALL_OUTCOMES:
                self.state.verifier_outcomes[key] = self.state.verifier_outcomes.get(
                    key, 0
                ) + selection.summary.count(key)
            self.logger.log(
                "rollouts",
                round=round_index,
                generated=len(rollouts),
                retained=len(selection.selected),
                retained_rate=selection.retained_rollout_rate,
                usable_rate=selection.summary.usable_rate,
                response_tokens=selection.response_tokens,
                outcomes=dict(selection.summary.counts),
                policy=selection.policy,
            )
            for micro in iter_micro_batches(
                selection.selected,
                micro_batch_size=self.config.micro_batch_size,
                max_tokens=self.config.max_micro_batch_tokens,
            ):
                self.train_token_batch(batch_builder(micro))
                if self.state.optimizer_steps >= self.config.max_optimizer_steps:
                    break
            if on_round_end is not None:
                on_round_end(round_index, self.state)
            if self.state.optimizer_steps >= self.config.max_optimizer_steps:
                break
        self.flush_gradients()
        return self.state

    def _default_batch_builder(
        self,
    ) -> Callable[[Sequence[Rollout]], Mapping[str, Any]]:
        tokenizer = self.student.tokenizer
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError(
                "The student tokenizer has no pad_token_id; a padded training "
                "batch cannot be built."
            )
        device = self._model_device(self.student)

        def build(selected: Sequence[Rollout]) -> Mapping[str, Any]:
            return build_token_batch(
                selected,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                device=device,
            )

        return build

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Save model/optimizer state and auditable trainer metadata."""

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError("Checkpointing requires PyTorch.") from exc
        checkpoint_path = (
            Path(path)
            if path is not None
            else Path(self.config.output_dir)
            / "checkpoints"
            / f"step-{self.state.optimizer_steps}.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model = getattr(self.student, "model", self.student)
        payload = {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "trainer_state": asdict(self.state),
            "trainer_config": asdict(self.config),
            "loss_config": asdict(self.loss_config),
        }
        torch.save(payload, checkpoint_path)
        self.logger.log("checkpoint", path=str(checkpoint_path), step=self.state.optimizer_steps)
        return checkpoint_path

    def _clip_and_step(self) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError("Training requires PyTorch.") from exc
        model = self.student.model
        if self._grad_scaler is not None:
            self._grad_scaler.unscale_(self.optimizer)
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config.max_grad_norm,
            )
        if self._grad_scaler is not None:
            self._grad_scaler.step(self.optimizer)
            self._grad_scaler.update()
        else:
            self.optimizer.step()
        if self._scheduler is not None:
            self._scheduler.step()
        self._zero_grad()
        self._window_loss_sum = 0.0
        self._window_tokens = 0.0

    def _zero_grad(self) -> None:
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def _configure_amp(self) -> None:
        """Resolve autocast and scaling after model devices are known.

        ``None`` means automatic behavior: use autocast only when CUDA is
        available and the configured/model dtype is BF16 or FP16.  CPU and
        float32 synthetic models therefore keep their ordinary eager path.
        """

        try:
            import torch
        except ImportError:  # pragma: no cover - training already needs torch
            return

        device = self._model_device(self.student)
        device_type = (
            str(device).split(":", maxsplit=1)[0]
            if device is not None
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        if device_type not in {"cuda", "cpu"}:
            self._disable_amp()
            return
        if device_type == "cuda" and not torch.cuda.is_available():
            self._disable_amp()
            return

        configured_dtype = self._configured_dtype()
        requested_dtype = self._normalize_dtype(self.config.amp_dtype)
        if requested_dtype is None:
            requested_dtype = configured_dtype
        if requested_dtype is None and self.config.amp_enabled is True:
            requested_dtype = (
                torch.bfloat16
                if device_type == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float16
                if device_type == "cuda"
                else torch.bfloat16
            )

        enabled = self.config.amp_enabled
        if enabled is None:
            enabled = (
                device_type == "cuda"
                and requested_dtype in {torch.bfloat16, torch.float16}
            )
        if not enabled or requested_dtype in {None, torch.float32}:
            self._disable_amp()
            return
        if device_type == "cpu" and requested_dtype == torch.float16:
            self._disable_amp()
            return
        if (
            device_type == "cuda"
            and requested_dtype == torch.bfloat16
            and not torch.cuda.is_bf16_supported()
        ):
            # An explicit request can still run safely with FP16 scaling.  The
            # automatic path disables AMP rather than silently changing dtype.
            if self.config.amp_enabled is None:
                self._disable_amp()
                return
            requested_dtype = torch.float16

        self._amp_enabled = True
        self._amp_dtype = requested_dtype
        self._amp_device_type = device_type
        use_scaler = (
            requested_dtype == torch.float16
            and device_type == "cuda"
            and self.config.amp_grad_scaler is not False
        )
        self._grad_scaler = self._make_grad_scaler(torch, use_scaler)

    def _configured_dtype(self) -> Any | None:
        """Find the configured dtype before falling back to parameter dtype."""

        try:
            import torch
        except ImportError:  # pragma: no cover - training already needs torch
            return None
        for wrapper in (self.student, self.teacher):
            options = getattr(wrapper, "options", None)
            configured = getattr(options, "dtype", None)
            normalized = self._normalize_dtype(configured)
            if normalized is not None:
                return normalized
        model = getattr(self.student, "model", None)
        if model is not None:
            try:
                return next(model.parameters()).dtype
            except (AttributeError, StopIteration):
                pass
        return torch.float32

    @staticmethod
    def _normalize_dtype(dtype: Any) -> Any | None:
        if dtype is None or dtype == "auto":
            return None
        try:
            import torch
        except ImportError:  # pragma: no cover - training already needs torch
            return None
        if dtype in {torch.bfloat16, torch.float16, torch.float32}:
            return dtype
        aliases = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        try:
            return aliases[str(dtype).lower()]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported AMP dtype {dtype!r}; use auto, bfloat16, "
                "float16, or float32."
            ) from exc

    @staticmethod
    def _dtype_name(dtype: Any | None) -> str | None:
        if dtype is None:
            return None
        return str(dtype).replace("torch.", "")

    def _disable_amp(self) -> None:
        self._amp_enabled = False
        self._amp_dtype = None
        self._amp_device_type = None
        self._grad_scaler = None

    @staticmethod
    def _make_grad_scaler(torch: Any, enabled: bool) -> Any | None:
        if not enabled:
            return None
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=True)

    def _autocast_context(self) -> Any:
        if not self._amp_enabled:
            return nullcontext()
        try:
            import torch
        except ImportError:  # pragma: no cover - training already needs torch
            return nullcontext()
        return torch.autocast(
            device_type=self._amp_device_type,
            dtype=self._amp_dtype,
            enabled=True,
        )

    def _freeze_teacher(self) -> None:
        """Keep the teacher in eval/inference mode even for generic wrappers."""

        try:
            model = self.teacher.model
        except (AttributeError, RuntimeError):
            return
        eval_model = getattr(model, "eval", None)
        if callable(eval_model):
            eval_model()
        freeze = getattr(model, "requires_grad_", None)
        if callable(freeze):
            freeze(False)

    @staticmethod
    def _model_device(wrapper: Any) -> Any | None:
        model = getattr(wrapper, "model", None)
        device = getattr(model, "device", None)
        if device is None and model is not None:
            try:
                device = next(model.parameters()).device
            except (AttributeError, StopIteration):
                pass
        return device

    @staticmethod
    def _extract_logits(outputs: Any) -> Any:
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, Mapping):
            logits = outputs.get("logits")
        if logits is None:
            raise ValueError("Model forward output does not contain logits.")
        return logits

    @staticmethod
    def _move_to_model_device(wrapper: Any, inputs: Mapping[str, Any]) -> dict[str, Any]:
        device = OPDTrainer._model_device(wrapper)
        if device is None:
            return dict(inputs)
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    @staticmethod
    def _no_grad() -> Any:
        try:
            import torch
        except ImportError:
            from contextlib import nullcontext

            return nullcontext()
        return torch.no_grad()


__all__ = ["OPDTrainer", "TrainerConfig", "TrainerState"]
