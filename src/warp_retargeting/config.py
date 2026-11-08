"""Typed configuration for the frozen WARP retargeting pipeline.

This is a package-oriented port of the paper checkpoint's schema and loader.
Defaults intentionally match the submitted converter configuration.
"""

from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import yaml


class ConfigError(ValueError):
    """Raised when a run configuration violates the frozen schema."""


@dataclass
class IOConfig:
    input_folder: str
    output_folder: str | None = None
    hdf5_name: str | None = None


@dataclass
class DataConfig:
    csv_reading_freq: float = 30.0
    no_filter: bool = False
    no_aria_black_image: bool = False
    task_id: int | None = None


@dataclass
class VideoConfig:
    no_video: bool = False
    per_demo_video: bool = False


@dataclass
class SEWSolverConfig:
    enable_collision_avoidance: bool = False
    enable_functional_retargeting: bool = True
    enable_functional_offset: bool = True
    enable_joint_limits: bool = False
    enable_stability_clamp: bool = False
    retargeting_mode: str | None = None
    # Preserve the frozen converter's misspelled public key for parity.
    base_alignement_mode: Literal["manual", "auto"] = "manual"
    mobile_base_spring_damper: bool = False
    mobile_base_deadband: float = 0.05
    mobile_base_yaw_deadband: float = 0.1
    mobile_base_natural_freq: float = 1.5
    mobile_base_damping_ratio: float = 1.0
    mobile_base_adaptive_freq: bool = False
    mobile_base_freq_min: float = 0.3
    mobile_base_adaptive_threshold: float = 0.15
    stability_radius: float = 0.5


@dataclass
class MinkSolverConfig:
    mink_max_iters: int = 40
    mink_dt: float = 1.0 / 60.0
    mink_damping: float = 0.5
    mink_pos_threshold: float = 1e-4
    mink_ori_threshold: float = 1e-4
    hand_position_cost: float = 1.0
    hand_orientation_cost: float = 1.0
    elbow_angle_cost: float = 0.0
    torso_orientation_cost: float = 0.5
    head_orientation_cost: float = 0.05
    upper_arm_direction_cost: float = 0.0
    lower_arm_direction_cost: float = 0.0
    posture_body_cost: float = 1e-2
    wrist_target_mode: Literal["palm", "wrist"] = "palm"
    soft_cost_start_deg: float = 2.0
    soft_cost_ramp_deg: float = 5.0
    enable_stability_clamp: bool = True
    stability_radius: float = 0.15
    mobile_base_alpha: float = 1.0


@dataclass
class RetargetConfig:
    solver: Literal["sew", "mink"]
    io: IOConfig
    enable_base_motion: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    sew: SEWSolverConfig | None = None
    mink: MinkSolverConfig | None = None

    def __post_init__(self) -> None:
        if self.solver not in ("sew", "mink"):
            raise ConfigError(f"solver must be 'sew' or 'mink', got {self.solver!r}")
        if self.solver == "sew":
            if self.mink is not None:
                raise ConfigError("mink section is invalid when solver='sew'")
            self.sew = self.sew or SEWSolverConfig()
        else:
            if self.sew is not None:
                raise ConfigError("sew section is invalid when solver='mink'")
            self.mink = self.mink or MinkSolverConfig()
        if self.io.hdf5_name is None:
            self.io.hdf5_name = "robot_data.hdf5" if self.solver == "sew" else "robot_data_mink.hdf5"


_SECTIONS: dict[str, type] = {
    "io": IOConfig,
    "data": DataConfig,
    "video": VideoConfig,
    "sew": SEWSolverConfig,
    "mink": MinkSolverConfig,
}


def _names(cls: type) -> set[str]:
    return {item.name for item in fields(cls)}


def _check_keys(raw: dict[str, Any], cls: type, context: str) -> None:
    unknown = set(raw) - _names(cls)
    if not unknown:
        return
    rendered = []
    for key in sorted(unknown):
        match = difflib.get_close_matches(key, _names(cls), n=1)
        rendered.append(f"{key!r} (did you mean {match[0]!r}?)" if match else repr(key))
    raise ConfigError(f"unknown key(s) in {context}: {', '.join(rendered)}")


def _section(raw: Any, cls: type, name: str):
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"{name}: expected mapping, got {type(raw).__name__}")
    _check_keys(raw, cls, name)
    try:
        return cls(**raw)
    except TypeError as exc:
        raise ConfigError(f"{name}: {exc}") from exc


def _resolve_io(cfg: RetargetConfig, anchor: Path) -> RetargetConfig:
    def resolve(value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else (anchor / path).resolve())

    return replace(cfg, io=replace(
        cfg.io,
        input_folder=resolve(cfg.io.input_folder),
        output_folder=resolve(cfg.io.output_folder),
    ))


def config_from_mapping(raw: dict[str, Any], anchor: str | Path | None = None) -> RetargetConfig:
    """Strictly validate a config mapping and optionally resolve its IO paths."""
    if not isinstance(raw, dict):
        raise ConfigError("top-level YAML must be a mapping")
    _check_keys(raw, RetargetConfig, "top level")
    if "solver" not in raw or "io" not in raw:
        raise ConfigError("config requires solver and io")
    solver = raw["solver"]
    cfg = RetargetConfig(
        solver=solver,
        io=_section(raw["io"], IOConfig, "io"),
        enable_base_motion=bool(raw.get("enable_base_motion", False)),
        data=_section(raw.get("data"), DataConfig, "data"),
        video=_section(raw.get("video"), VideoConfig, "video"),
        sew=_section(raw.get("sew"), SEWSolverConfig, "sew") if "sew" in raw else None,
        mink=_section(raw.get("mink"), MinkSolverConfig, "mink") if "mink" in raw else None,
    )
    if cfg.sew and cfg.sew.retargeting_mode not in (None, "pose", "tcp", "left_elbow", "right_elbow"):
        raise ConfigError(f"invalid sew.retargeting_mode: {cfg.sew.retargeting_mode!r}")
    if cfg.mink and cfg.mink.wrist_target_mode not in ("palm", "wrist"):
        raise ConfigError(f"invalid mink.wrist_target_mode: {cfg.mink.wrist_target_mode!r}")
    return _resolve_io(cfg, Path(anchor).resolve()) if anchor is not None else cfg


def load_config(path: str | Path) -> RetargetConfig:
    """Load, strictly validate, and path-resolve a frozen run config."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"failed to parse YAML {config_path}: {exc}") from exc
    return config_from_mapping(raw, config_path.parent)


def dump_run_config(cfg: RetargetConfig, path: str | Path) -> None:
    """Write all effective fields while omitting the inactive solver block."""
    payload = asdict(cfg)
    payload.pop("mink" if cfg.solver == "sew" else "sew")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False))
