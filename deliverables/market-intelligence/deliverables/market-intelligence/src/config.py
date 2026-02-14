"""config.py — Load YAML configuration.

A single YAML file drives paths, thresholds, and model settings so nothing
is hard-coded in the pipeline modules.
"""
from __future__ import annotations

from typing import Any, Dict

import yaml


def load_config(path: str = "configs/v1.yaml") -> Dict[str, Any]:
    """Load YAML config from *path*.

    Raises a clear error if the file cannot be read.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Run from repo root or pass --config <path>."
        ) from e

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config format in {path}: expected a YAML mapping.")
    return cfg
