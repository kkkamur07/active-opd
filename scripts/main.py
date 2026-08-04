"""Hydra entry point for configuration inspection and future experiments."""

from __future__ import annotations

from typing import Any

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # pragma: no cover - exercised only in minimal environments
    hydra = None
    DictConfig = Any
    OmegaConf = None


if hydra is not None:

    @hydra.main(
        version_base=None,
        config_path="../configs",
        config_name="config",
    )
    def main(config: DictConfig) -> None:
        """Resolve and print config without loading models or datasets."""

        print(OmegaConf.to_yaml(config, resolve=True))

else:

    def main(config: Any | None = None) -> None:
        """Explain the optional runtime dependency when Hydra is unavailable."""

        raise ImportError(
            "The Hydra CLI requires the project dependencies; install hydra-core."
        )


if __name__ == "__main__":
    main()
