"""Dependency-light frozen WARP HDF5 benchmark metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .hdf5 import list_demos


R_HUMAN_ROBOT = {
    "right": np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    "left": np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
}
P_WRIST_TO_PALM = {
    # Exact XHand index-MCP/palm target in RBY1 joint-7 coordinates. These
    # values are derived from the public URDF/MJCF chain, not solver source.
    "right": np.array([-0.0065, 0.0265, -0.2097]),
    "left": np.array([-0.0065, -0.0265, -0.2097]),
}


def _rotation_error_deg(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.einsum("tji,tjk->tik", actual, target)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def _nan_stats(values: np.ndarray, skip: int = 0) -> dict[str, float]:
    values = np.asarray(values, dtype=float)[skip:]
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def _trajectory_derivative(
    q: np.ndarray, *, order: int, dt: float, skip: int = 0
) -> np.ndarray:
    """Paper-style forward difference, aligned to the first frame of a window."""
    result = np.full((len(q), q.shape[1]), np.nan, dtype=float)
    if len(q) <= order:
        return result
    result[: len(q) - order] = np.diff(q, n=order, axis=0) / (dt ** order)
    if skip:
        result[: min(skip, len(result))] = np.nan
    return result


def compute_metrics(hdf5_file: str | Path, demo_key: str = "demo_0") -> tuple[dict, dict]:
    """Compute the portable subset of the frozen RBY1 benchmark vocabulary."""
    with h5py.File(hdf5_file, "r") as root:
        demo = root["data"][demo_key]
        obs = demo["obs"]
        targets = demo["targets"]
        transforms = demo["transforms"]
        q = np.asarray(obs["robot0_joint_pos"][:], dtype=float)
        t = np.asarray(obs["robot_ts"][:], dtype=float)
        fps = float(demo.attrs.get("fps", 30.0))
        R_mocap_world = np.asarray(transforms["R_mocap_world"][:], dtype=float)
        p_mocap_world = np.asarray(transforms["p_mocap_world"][:], dtype=float)
        contacts = np.asarray(demo["diagnostics/self_contacts"][:], dtype=int)
        per_frame: dict[str, np.ndarray] = {}

        for side in ("right", "left"):
            sew = np.asarray(targets[f"human_{side}_sew"][:], dtype=float)
            upper_R = np.asarray(targets["human_upper_rotation"][:], dtype=float)
            upper_p = np.asarray(targets["human_upper_position"][:], dtype=float)
            mcp = np.asarray(targets[f"{side}_mcp_centroid"][:], dtype=float)
            wrist = sew[:, 6:9]
            wrist_R = sew[:, 9:18].reshape(-1, 3, 3)
            palm_upper = wrist + np.einsum("tij,tj->ti", wrist_R, mcp)
            palm_mocap = upper_p + np.einsum("tij,tj->ti", upper_R, palm_upper)
            palm_target = (R_mocap_world.T @ (palm_mocap - p_mocap_world).T).T
            wrist_target_R = np.einsum(
                "ij,tjk,tkl,lm->tim",
                R_mocap_world.T, upper_R, wrist_R, R_HUMAN_ROBOT[side],
            )
            wrist_robot_T = np.asarray(targets[f"wrist_{side}"][:], dtype=float).reshape(-1, 4, 4)
            palm_robot = wrist_robot_T[:, :3, 3] + np.einsum(
                "tij,j->ti", wrist_robot_T[:, :3, :3], P_WRIST_TO_PALM[side]
            )
            per_frame[f"eef_position_{side}_m"] = np.linalg.norm(
                palm_robot - palm_target, axis=1
            )
            per_frame[f"eef_orientation_{side}_deg"] = _rotation_error_deg(
                wrist_robot_T[:, :3, :3], wrist_target_R
            )

        human_torso_R = np.einsum(
            "ij,tjk->tik", R_mocap_world.T,
            np.asarray(targets["human_upper_rotation"][:], dtype=float),
        )
        robot_torso_R = np.asarray(targets["robot_torso"][:], dtype=float).reshape(-1, 4, 4)[:, :3, :3]
        per_frame["torso_orientation_deg"] = _rotation_error_deg(robot_torso_R, human_torso_R)

        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / fps
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0 / fps
        q_unwrapped = q.copy()
        for index in (10, 12, 14, 17, 19, 21):
            q_unwrapped[:, index] = np.unwrap(q_unwrapped[:, index])
        q_body = q_unwrapped[:, 4:26]
        velocity = _trajectory_derivative(q_body, order=1, dt=dt)
        jerk = _trajectory_derivative(q_body, order=3, dt=dt, skip=5)
        per_frame["joint_velocity_l2_rad_s"] = np.linalg.norm(velocity, axis=1)
        per_frame["joint_jerk_l2_rad_s3"] = np.linalg.norm(jerk, axis=1)
        per_frame["self_collision"] = contacts.astype(bool)

    summary = {
        "hdf5": str(hdf5_file),
        "demo_key": demo_key,
        "frames": int(len(q)),
        "fps": fps,
        "eef_position_right_mm": {
            key: value * 1000.0 for key, value in _nan_stats(per_frame["eef_position_right_m"], 5).items()
        },
        "eef_position_left_mm": {
            key: value * 1000.0 for key, value in _nan_stats(per_frame["eef_position_left_m"], 5).items()
        },
        "eef_orientation_right_deg": _nan_stats(per_frame["eef_orientation_right_deg"], 5),
        "eef_orientation_left_deg": _nan_stats(per_frame["eef_orientation_left_deg"], 5),
        "torso_orientation_deg": _nan_stats(per_frame["torso_orientation_deg"], 5),
        "joint_velocity_l2_rad_s": _nan_stats(per_frame["joint_velocity_l2_rad_s"]),
        "joint_jerk_l2_rad_s3": _nan_stats(per_frame["joint_jerk_l2_rad_s3"]),
        "self_collision_frame_fraction": float(
            np.count_nonzero(contacts[5:]) / max(1, len(contacts[5:]))
        ),
    }
    return summary, per_frame


def _aggregate(summaries: list[dict]) -> dict:
    scalar_paths = (
        "eef_position_right_mm", "eef_position_left_mm",
        "eef_orientation_right_deg", "eef_orientation_left_deg",
        "torso_orientation_deg", "joint_velocity_l2_rad_s",
        "joint_jerk_l2_rad_s3",
    )
    aggregate = {
        key: {
            statistic: float(np.nanmean([
                item[key][statistic] for item in summaries
            ]))
            for statistic in ("mean", "median", "max")
        }
        for key in scalar_paths
    }
    aggregate["self_collision_frame_fraction"] = float(np.nanmean([
        item["self_collision_frame_fraction"] for item in summaries
    ]))
    return aggregate


def write_metrics(
    hdf5_file, output_dir, demo_key: str | None = "demo_0"
) -> dict:
    keys = list(list_demos(hdf5_file)) if demo_key is None else [demo_key]
    if not keys:
        raise ValueError(f"no demos found in {hdf5_file}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    with h5py.File(output_dir / "metrics.h5", "w") as root:
        for key in keys:
            summary, per_frame = compute_metrics(hdf5_file, key)
            summaries.append(summary)
            demo = root.create_group(key)
            for metric, values in per_frame.items():
                demo.create_dataset(metric, data=values)
    result = summaries[0] if len(summaries) == 1 else {
        "hdf5": str(hdf5_file),
        "demos": len(summaries),
        "frames": int(sum(item["frames"] for item in summaries)),
        "aggregate": _aggregate(summaries),
        "per_demo": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5_file")
    parser.add_argument("--demo-key", default="demo_0")
    parser.add_argument("--all-demos", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output = args.output_dir or str(Path(args.hdf5_file).with_suffix("")) + "_metrics"
    summary = write_metrics(
        args.hdf5_file, output, None if args.all_demos else args.demo_key
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
