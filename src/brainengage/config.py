from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Load configs/default.yaml (or the given file). All relative paths in the
    config are relative to the repo root; use :func:`resolve` to make them absolute."""
    cfg_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
