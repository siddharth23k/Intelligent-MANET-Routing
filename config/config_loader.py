"""Typed access to paper_scenarios.yaml.

Every value both methods must agree on is read through here, so a parameter
cannot be defined twice with two different numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

DEFAULT_CONFIG = Path(__file__).parent / "paper_scenarios.yaml"


class Config:
    def __init__(self, config_path: str | Path | None = None):
        with open(config_path or DEFAULT_CONFIG, "r", encoding="utf-8") as handle:
            self.config: Dict[str, Any] = yaml.safe_load(handle)

    # --- scenario geometry -------------------------------------------------
    @property
    def area_size(self) -> List[int]:
        return self.config["paper_reference"]["area_m"]

    @property
    def area_center(self) -> float:
        return self.area_size[0] / 2.0

    @property
    def communication_radius_default(self) -> float:
        return float(self.config["evaluation_defaults"]["radius_m"])

    @property
    def communication_radii(self) -> List[float]:
        return [float(r) for r in self.config["paper_reference"]["communication_radius_m"]]

    # --- evaluation --------------------------------------------------------
    @property
    def random_seed(self) -> int:
        return int(self.config["evaluation_defaults"]["random_seed"])

    @property
    def test_run_count(self) -> int:
        return int(self.config["evaluation_defaults"]["test_run_count"])

    @property
    def val_run_count(self) -> int:
        return int(self.config["evaluation_defaults"]["val_run_count"])

    @property
    def pairs_per_step(self) -> int:
        return int(self.config["evaluation_defaults"]["pairs_per_step"])

    @property
    def hello_interval(self) -> float:
        return float(self.config["evaluation_defaults"]["hello_interval_s"])

    # --- labels ------------------------------------------------------------
    @property
    def labels(self) -> Dict[str, Any]:
        return self.config["labels"]

    @property
    def label_horizon(self) -> int:
        return int(self.config["labels"]["horizon_steps"])

    @property
    def rssi_sentinel(self) -> float:
        return float(self.config["labels"]["rssi_sentinel"])

    @property
    def rssi_floor(self) -> float:
        return float(self.config["labels"]["rssi_floor_db"])

    # --- metrics and models ------------------------------------------------
    @property
    def survival_horizon(self) -> int:
        return int(self.config["survival"]["horizon_steps"])

    @property
    def training(self) -> Dict[str, Any]:
        return self.config["training"]

    @property
    def smoke(self) -> Dict[str, Any]:
        return self.config["smoke"]

    def get_paper_reference_config(self) -> Dict[str, Any]:
        return self.config["paper_reference"]

    def get_evaluation_defaults(self) -> Dict[str, Any]:
        return self.config["evaluation_defaults"]


_config: Config | None = None


def get_config() -> Config:
    """Process wide singleton."""
    global _config
    if _config is None:
        _config = Config()
    return _config
