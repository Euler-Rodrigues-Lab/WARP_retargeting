"""Frozen RBY1 robomimic-HDF5 layout used by the WARP paper pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


SCHEMA = "warp.rby1.robomimic/1"
ACTION_DIM = 49

ACT_LEFT_ARM = slice(0, 7)
ACT_RIGHT_ARM = slice(7, 14)
ACT_TORSO = slice(14, 20)
ACT_HEAD = slice(20, 22)
ACT_BASE = slice(22, 25)
ACT_LEFT_HAND = slice(25, 37)
ACT_RIGHT_HAND = slice(37, 49)

OBS_TORSO = slice(4, 10)
OBS_RIGHT_ARM = slice(10, 17)
OBS_LEFT_ARM = slice(17, 24)
OBS_HEAD = slice(24, 26)


def _f32(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _require_rows(arrays: dict[str, np.ndarray]) -> int:
    lengths = {key: len(np.asarray(value)) for key, value in arrays.items()}
    if not lengths:
        raise ValueError("cannot write an empty episode")
    unique = set(lengths.values())
    if len(unique) != 1:
        raise ValueError(f"episode arrays have mismatched row counts: {lengths}")
    n = unique.pop()
    if n <= 0:
        raise ValueError("cannot write an empty episode")
    return n


def build_action_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build the exact frozen action groups from named trajectory arrays."""
    joint = np.hstack([
        arrays["q_left"], arrays["q_right"], arrays["q_torso"], arrays["q_head"],
        arrays["base_delta"], arrays["q_left_hand"], arrays["q_right_hand"],
    ])
    if joint.shape[1] != ACTION_DIM:
        raise ValueError(f"actions/joint must have {ACTION_DIM} columns, got {joint.shape}")
    return {
        "joint": _f32(joint),
        "joint_arm": _f32(np.hstack([arrays["q_left"], arrays["q_right"]])),
        "joint_arm_hand": _f32(np.hstack([
            arrays["q_left"], arrays["q_right"],
            arrays["q_left_hand"], arrays["q_right_hand"],
        ])),
        "joint_arm_head_torso_base": _f32(np.hstack([
            arrays["q_left"], arrays["q_right"], arrays["q_torso"],
            arrays["q_head"], arrays["base_delta"],
        ])),
        "eef": _f32(np.hstack([
            arrays["eef_right"], arrays["eef_left"],
            arrays["q_left_hand"], arrays["q_right_hand"],
        ])),
    }


def write_episode(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    *,
    demo_key: str = "demo_0",
    fps: float = 30.0,
    source: str = "",
    variant: str = "",
    config: dict | None = None,
) -> Path:
    """Write one frozen WARP episode in its original robomimic-style schema."""
    required = {
        "q_left", "q_right", "q_torso", "q_head", "q_left_hand",
        "q_right_hand", "base_delta", "base_position", "base_rotation",
        "eef_left", "eef_right", "solve_ms", "self_contacts",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise KeyError(f"missing episode arrays: {', '.join(missing)}")
    n = _require_rows({key: arrays[key] for key in required})
    actions = build_action_arrays(arrays)

    robot_pos = np.hstack([
        np.zeros((n, 4), dtype=np.float32), arrays["q_torso"],
        arrays["q_right"], arrays["q_left"], arrays["q_head"],
    ]).astype(np.float32)
    timestamp = np.arange(n, dtype=np.float64) / float(fps)
    robot_vel = np.zeros_like(robot_pos)
    if n > 1:
        robot_vel[1:] = np.diff(robot_pos, axis=0) * float(fps)
        robot_vel[0] = robot_vel[1]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as root:
        data = root.require_group("data")
        data.attrs.setdefault("env_args", json.dumps({
            "env_name": "RBY1Teleop", "env_type": "robosuite", "env_kwargs": {},
        }))
        if demo_key in data:
            del data[demo_key]
        demo = data.create_group(demo_key)
        demo.attrs["num_samples"] = n
        demo.attrs["schema"] = SCHEMA
        demo.attrs["source"] = str(source)
        demo.attrs["variant"] = str(variant)
        demo.attrs["fps"] = float(fps)
        if config is not None:
            demo.attrs["run_config"] = json.dumps(config, sort_keys=True)

        obs = demo.create_group("obs")
        obs.create_dataset("robot0_joint_pos", data=robot_pos)
        obs.create_dataset("robot0_joint_vel", data=robot_vel)
        obs.create_dataset("robot_ts", data=timestamp)
        obs.create_dataset("cmd_ts", data=timestamp)
        for side in ("left", "right"):
            hand = _f32(arrays[f"q_{side}_hand"])
            obs.create_dataset(f"hand_{side}_qpos", data=hand)
            obs.create_dataset(f"hand_{side}_cmd_qpos", data=hand)
            obs.create_dataset(f"hand_ts_{side}", data=timestamp)
        obs.create_dataset("eef_left_proprio", data=_f32(arrays["eef_left"]))
        obs.create_dataset("eef_right_proprio", data=_f32(arrays["eef_right"]))

        action_group = demo.create_group("actions")
        for key, value in actions.items():
            action_group.create_dataset(key, data=value)

        transforms = demo.create_group("transforms")
        transforms.create_dataset("p_world_base", data=np.asarray(arrays["base_position"], dtype=np.float64))
        transforms.create_dataset("R_world_base", data=np.asarray(arrays["base_rotation"], dtype=np.float64))
        if "mocap_world_position" in arrays:
            transforms.create_dataset("p_mocap_world", data=np.asarray(arrays["mocap_world_position"], dtype=np.float64))
        if "mocap_world_rotation" in arrays:
            transforms.create_dataset("R_mocap_world", data=np.asarray(arrays["mocap_world_rotation"], dtype=np.float64))

        targets = demo.create_group("targets")
        for key in (
            "human_left_sew", "human_right_sew", "human_upper_rotation",
            "human_upper_position", "left_mcp_centroid", "right_mcp_centroid",
            "palm_left", "palm_right", "wrist_left", "wrist_right", "robot_torso",
        ):
            if key in arrays:
                targets.create_dataset(key, data=np.asarray(arrays[key], dtype=np.float64))

        diagnostics = demo.create_group("diagnostics")
        diagnostics.create_dataset("solve_time_s", data=np.asarray(arrays["solve_ms"], dtype=np.float64) / 1000.0)
        diagnostics.create_dataset("self_contacts", data=np.asarray(arrays["self_contacts"], dtype=np.int64))
        demo.create_dataset("rewards", data=np.zeros(n, dtype=np.float32))
        demo.create_dataset("dones", data=np.zeros(n, dtype=np.float32))
        data.attrs["total"] = int(sum(int(data[key].attrs["num_samples"]) for key in data))
    return path


def list_demos(path: str | Path) -> tuple[str, ...]:
    with h5py.File(path, "r") as root:
        keys = [key for key in root["data"] if key.startswith("demo_")]
        return tuple(sorted(keys, key=lambda key: int(key.removeprefix("demo_"))))


def load_episode(path: str | Path, demo_key: str = "demo_0") -> dict[str, np.ndarray | float | str]:
    """Load the fields needed by replay, policy-dataset inspection, and metrics."""
    with h5py.File(path, "r") as root:
        demo = root["data"][demo_key]
        obs = demo["obs"]
        result: dict[str, np.ndarray | float | str] = {
            "robot_pos": obs["robot0_joint_pos"][:],
            "robot_ts": obs["robot_ts"][:],
            "q_left_hand": obs["hand_left_qpos"][:],
            "q_right_hand": obs["hand_right_qpos"][:],
            "actions_joint": demo["actions"]["joint"][:],
            "fps": float(demo.attrs.get("fps", 30.0)),
            "variant": str(demo.attrs.get("variant", "")),
            "source": str(demo.attrs.get("source", "")),
        }
        if "transforms" in demo:
            result["base_position"] = demo["transforms"]["p_world_base"][:]
            result["base_rotation"] = demo["transforms"]["R_world_base"][:]
        return result
