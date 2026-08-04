from pathlib import Path

import pytest

hydra = pytest.importorskip("hydra")
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from aopd.losses import OPDLossConfig
from aopd.train import TrainerConfig
from aopd.train.rollouts import RolloutCollectionConfig

CONFIG_DIR = Path(__file__).parents[1] / "configs"


def _compose(*overrides):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=list(overrides))


def test_default_hydra_config_resolves_experiment_controls():
    config = _compose()

    assert config.model.student.model_id == "Qwen/Qwen3-1.7B"
    assert config.rollout.num_rollouts_per_prompt == 8
    assert config.generation.enable_thinking is True
    # A real thinking budget: at 1024 every math trace was cut off mid-<think>
    # and the verifier outcome became a length statistic.
    assert config.generation.max_new_tokens >= 8192
    assert config.rollout.max_new_tokens == config.generation.max_new_tokens
    assert config.estimator.estimator == "exact_reverse_kl"
    assert config.trainer.master_weights == "fp32"


def test_default_training_corpus_has_a_ground_truth_answer_column():
    """OpenThoughts-114k has none, and its first ~21k records are codegen."""

    config = _compose()
    assert "openthoughts" not in config.data.dataset_name.lower()


def test_config_groups_map_onto_the_dataclasses_that_consume_them():
    """Every YAML key must be consumed. `from_mapping` now raises on unknown
    keys, so a silently-ignored setting fails here rather than in a run."""

    config = _compose()
    container = OmegaConf.to_container(config, resolve=True)

    loss = OPDLossConfig.from_mapping(container["estimator"])
    assert loss.estimator == "exact_reverse_kl"

    trainer = TrainerConfig.from_mapping(container["trainer"])
    assert trainer.gradient_accumulation_steps == container["precision"][
        "gradient_accumulation_steps"
    ]
    assert trainer.micro_batch_size == container["precision"]["micro_batch_size"]

    rollout = RolloutCollectionConfig.from_mapping(container["rollout"])
    assert rollout.num_rollouts_per_prompt == 8
    assert rollout.retain_token_ids is True


def test_filtering_groups_select_the_documented_policies():
    for group, policy in (
        ("verified_wrong", "verified_wrong"),
        ("all", "all"),
        ("random", "random"),
    ):
        config = _compose(f"filtering={group}")
        assert config.filtering.policy == policy


def test_each_run_gets_its_own_output_directory():
    """Runs previously shared one dir, so metrics.jsonl accumulated several
    invocations into a single unseparable curve."""

    config = _compose()
    assert "${run_name}" in OmegaConf.to_yaml(config.paths, resolve=False) or (
        config.run_name in config.paths.run_dir
    )
