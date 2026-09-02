"""Frozen paper-variant registry loading."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from .config import ConfigError, RetargetConfig, config_from_mapping


DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[2]
    / "experiments/warp_paper/configs/variants.yaml"
)


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def variant_names(registry: str | Path = DEFAULT_REGISTRY) -> tuple[str, ...]:
    raw = yaml.safe_load(Path(registry).read_text())
    return tuple(raw.get("variants", {}))


def load_variant(
    name: str,
    *,
    input_folder: str,
    output_folder: str | None = None,
    registry: str | Path = DEFAULT_REGISTRY,
) -> tuple[RetargetConfig, str]:
    """Materialize one frozen variant as a validated runtime config."""
    path = Path(registry)
    raw = yaml.safe_load(path.read_text())
    variants = raw.get("variants", {})
    if name not in variants:
        raise ConfigError(
            f"unknown variant {name!r}; available: {', '.join(variants)}"
        )
    selected = deepcopy(variants[name])
    description = str(selected.pop("description", name))
    merged = _deep_merge(raw.get("defaults", {}), selected)
    merged["io"] = {
        "input_folder": input_folder,
        "output_folder": output_folder,
    }
    return config_from_mapping(merged), description
