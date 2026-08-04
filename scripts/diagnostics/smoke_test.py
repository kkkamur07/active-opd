"""Opt-in, one-step smoke test for the configured Qwen3 wrappers.

This module is intentionally inert unless invoked with ``--run``.  Importing
it, opening the notebook, or running it without that flag never loads a model
or contacts the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Opt into loading Qwen3-1.7B and Qwen3-4B.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Short generation budget for the smoke test.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/real-model-smoke",
        help="Directory for metrics written by OPDTrainer.",
    )
    return parser.parse_args()


def _resolved_config(config_dir: Path) -> Any:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="config")
    return OmegaConf.to_container(config, resolve=True)


def run_smoke(*, max_new_tokens: int, output_dir: str) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch

    if not torch.cuda.is_available():
        print(
            "REAL_MODEL_SMOKE_BLOCKED: CUDA is unavailable; loading the "
            "configured 1.7B/4B pair on CPU is not a short smoke test.",
            file=sys.stderr,
        )
        return 2
    torch.cuda.set_device(0)
    print(
        "cuda:",
        {
            "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "device": torch.cuda.get_device_name(0),
            "torch_cuda": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
    )

    def nvidia_smi() -> str:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def memory() -> dict[str, int]:
        return {
            "allocated": int(torch.cuda.memory_allocated()),
            "reserved": int(torch.cuda.memory_reserved()),
            "max_allocated": int(torch.cuda.max_memory_allocated()),
        }

    from aopd.data import Rollout
    from aopd.losses import OPDLossConfig
    from aopd.models import GenerationOptions, StudentModel, TeacherModel
    from aopd.train import OPDTrainer, RolloutCollector, TrainerConfig

    repo_root = Path(__file__).resolve().parent.parent
    resolved = _resolved_config(repo_root / "configs")
    student = None
    teacher = None
    try:
        student = StudentModel.from_config(resolved["model"]["student"])
        teacher = TeacherModel.from_config(resolved["model"]["teacher"])
        trainer_values = dict(resolved["trainer"])
        trainer_values.update(
            {
                "max_steps": 1,
                "gradient_accumulation_steps": 1,
                "use_8bit_optimizer": False,
                "checkpoint_every": 0,
                "output_dir": output_dir,
                "amp_enabled": True,
                "amp_dtype": resolved["precision"]["compute_dtype"],
            }
        )
        trainer = OPDTrainer(
            student,
            teacher,
            config=TrainerConfig.from_mapping(trainer_values),
            loss_config=OPDLossConfig.from_mapping(resolved["estimator"]),
        )
        trainer.initialize()
        student.assert_tokenizer_compatible(teacher)
        student_devices = {str(parameter.device) for parameter in student.model.parameters()}
        teacher_devices = {str(parameter.device) for parameter in teacher.model.parameters()}
        teacher_map = getattr(teacher.model, "hf_device_map", None)
        if not student_devices or any(
            not device.startswith("cuda") for device in student_devices
        ):
            raise RuntimeError(f"Student was not fully placed on CUDA: {student_devices}")
        if not teacher_devices or any(
            not device.startswith("cuda") for device in teacher_devices
        ):
            raise RuntimeError(
                f"Teacher was not fully placed on CUDA: devices={teacher_devices}, "
                f"device_map={teacher_map}"
            )
        if not trainer.amp_enabled or trainer.active_amp_dtype != "bfloat16":
            raise RuntimeError(
                "Expected CUDA BF16 autocast, got "
                f"enabled={trainer.amp_enabled}, dtype={trainer.active_amp_dtype}"
            )
        print(
            "placement:",
            {
                "student_devices": sorted(student_devices),
                "teacher_devices": sorted(teacher_devices),
                "teacher_device_map": teacher_map,
                "amp_enabled": trainer.amp_enabled,
                "amp_dtype": trainer.active_amp_dtype,
            },
        )
        torch.cuda.reset_peak_memory_stats()
        print("telemetry_before:", {"memory": memory(), "nvidia_smi": nvidia_smi()})

        prompt = "Solve this exactly and end with \\boxed{answer}: What is 2 + 2?"
        generation_values = dict(resolved["generation"])
        generation_values.update(
            {
                "max_new_tokens": max_new_tokens,
                "num_return_sequences": 1,
                "do_sample": False,
            }
        )
        generation = GenerationOptions.from_mapping(generation_values)
        collector = RolloutCollector(
            student,
            config={
                "num_rollouts_per_prompt": 1,
                "max_new_tokens": max_new_tokens,
            },
        )
        rollouts = collector.collect([prompt], generation=generation)
        if not rollouts or not isinstance(rollouts[0], Rollout):
            raise RuntimeError("The student did not return a tokenized rollout.")
        rollout = rollouts[0]
        if rollout.input_ids is None or rollout.prompt_length is None:
            raise RuntimeError("The rollout did not retain prompt/response tokens.")

        input_ids = torch.as_tensor(rollout.input_ids)
        model_device = getattr(student.model, "device", None)
        if model_device is not None:
            input_ids = input_ids.to(model_device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "prompt_lengths": torch.tensor(
                [rollout.prompt_length],
                device=input_ids.device,
            ),
        }
        metrics = trainer.train_token_batch(batch)
        print("telemetry_after:", {"memory": memory(), "nvidia_smi": nvidia_smi()})
        print("REAL_MODEL_SMOKE_OK")
        print(f"student: {student.options.model_id}")
        print(f"teacher: {teacher.options.model_id}")
        print(f"response preview: {rollout.response[:240]!r}")
        print(
            "trainer:",
            {
                "loss": metrics["loss"],
                "amp_enabled": trainer.amp_enabled,
                "amp_dtype": trainer.active_amp_dtype,
                "grad_scaler": trainer.grad_scaler_enabled,
            },
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - report blocked smoke-test failures
        print(
            f"REAL_MODEL_SMOKE_BLOCKED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    finally:
        for wrapper in (student, teacher):
            if wrapper is not None:
                wrapper.unload()


def main() -> int:
    args = _parse_args()
    if not args.run:
        print(
            "Real-model smoke is disabled. Re-run with --run to explicitly "
            "load the configured Qwen3 checkpoints."
        )
        return 0
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive.")
    return run_smoke(
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
