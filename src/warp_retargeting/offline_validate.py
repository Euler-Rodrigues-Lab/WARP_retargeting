"""Deterministic offline validation for frozen WARP and MINK variants."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from geo_kin_core.frames import load_frames
from geo_kin_core.session import resolve_session
from rby1_teleop import SAMPLE_MOTION, SPECS_DIR, XML_RBY1_XHAND
from rby1_teleop.control import RBY1WithXHandMuJoCoController

from .hdf5 import write_episode
from .variants import DEFAULT_REGISTRY, load_variant, variant_names


def _self_contacts(model, data) -> int:
    count = 0
    for i in range(data.ncon):
        contact = data.contact[i]
        body_a = model.geom_bodyid[contact.geom1]
        body_b = model.geom_bodyid[contact.geom2]
        if body_a != 0 and body_b != 0 and body_a != body_b:
            count += 1
    return count


def _body_transform(model, data, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        return np.zeros(16, dtype=np.float32)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = data.xmat[body_id].reshape(3, 3)
    transform[:3, 3] = data.xpos[body_id]
    return transform.reshape(16)


def _optional_array(value, shape) -> np.ndarray:
    if value is None:
        return np.full(shape, np.nan, dtype=float)
    return np.asarray(value, dtype=float).reshape(shape)


def _base_deltas(positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """Absolute world base poses -> frozen world/frame-0-heading SE(2) deltas."""
    positions = np.asarray(positions, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    relative_rotation = rotations[0].T @ rotations
    relative_position = (rotations[0].T @ (positions - positions[0]).T).T
    absolute = np.column_stack([
        relative_position[:, 0],
        relative_position[:, 1],
        np.arctan2(relative_rotation[:, 1, 0], relative_rotation[:, 0, 0]),
    ])
    deltas = np.zeros_like(absolute)
    deltas[0] = absolute[0]
    deltas[1:] = np.diff(absolute, axis=0)
    deltas[:, 2] = (deltas[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return deltas.astype(np.float32)


def _normalize_goals(out, controller) -> None:
    """Select the nearest 2π branch, matching the frozen CSV converter."""
    for name, current in (
        ("q_goal_right", controller.q_current_right),
        ("q_goal_left", controller.q_current_left),
        ("q_goal_torso", controller.q_current_torso),
        ("q_goal_head", controller.q_current_head),
    ):
        value = getattr(out, name, None)
        if value is not None:
            setattr(out, name, controller.real_angle(np.asarray(value), current))


def _session(cfg, rate_hz: float):
    if cfg.solver == "mink":
        kwargs = asdict(cfg.mink)
        return resolve_session(
            "rby1",
            hand="xhand",
            backend="mink",
            model_xml=XML_RBY1_XHAND,
            mobile_base=cfg.enable_base_motion,
            **kwargs,
        )

    try:
        import geo_kin  # noqa: F401 - fail early with an actionable message
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SEW variants require the licensed geo_kin wheel. Install it into "
            "this environment after `uv sync`, for example: "
            "`uv pip install --python .venv/bin/python /path/to/geo_kin.whl`."
        ) from exc

    sew = cfg.sew
    mode = sew.retargeting_mode
    if mode is None:
        mode = "tcp" if sew.enable_functional_retargeting else "pose"
    try:
        return resolve_session(
            "rby1",
            hand="xhand",
            backend="licensed",
            spec_dir=SPECS_DIR,
            control_rate_hz=rate_hz,
            retarget_mode=mode,
            collision_avoidance=sew.enable_collision_avoidance,
            functional_offset=sew.enable_functional_offset,
            torso_ik_limited=sew.enable_joint_limits,
            arm_ik_limited=sew.enable_joint_limits,
            mobile_base=cfg.enable_base_motion,
            base_alignment_mode=sew.base_alignement_mode,
            mobile_base_spring_damper=sew.mobile_base_spring_damper,
            mobile_base_deadband=sew.mobile_base_deadband,
            mobile_base_yaw_deadband=sew.mobile_base_yaw_deadband,
            mobile_base_natural_freq=sew.mobile_base_natural_freq,
            mobile_base_damping_ratio=sew.mobile_base_damping_ratio,
            mobile_base_adaptive_freq=sew.mobile_base_adaptive_freq,
            mobile_base_freq_min=sew.mobile_base_freq_min,
            mobile_base_adaptive_threshold=sew.mobile_base_adaptive_threshold,
            enable_stability_clamp=sew.enable_stability_clamp,
            stability_radius=sew.stability_radius,
        )
    except TypeError as exc:
        raise RuntimeError(
            "This frozen variant needs the RBY1 mobile-base configuration API "
            "added in geo_kin@9dbd4ed. Install a wheel built from that commit "
            "or newer."
        ) from exc


def run_validation(
    variant: str,
    *,
    frames: str | Path = SAMPLE_MOTION,
    output_dir: str | Path,
    max_frames: int | None = None,
    registry: str | Path = DEFAULT_REGISTRY,
    demo_key: str = "demo_0",
    hdf5_path: str | Path | None = None,
) -> dict:
    """Replay a frame stream through one frozen variant and save diagnostics."""
    frames = Path(frames)
    output_dir = Path(output_dir)
    cfg, description = load_variant(
        variant,
        input_folder=str(frames.parent),
        output_folder=str(output_dir),
        registry=registry,
    )
    stream = load_frames(frames)
    count = len(stream) if max_frames is None else min(len(stream), int(max_frames))
    if count <= 0:
        raise ValueError("validation needs at least one frame")

    model = mujoco.MjModel.from_xml_path(str(XML_RBY1_XHAND))
    data = mujoco.MjData(model)
    controller = RBY1WithXHandMuJoCoController(model, data, debug=False)
    controller.setup_mocap_body("base_mocap_mover")
    session = _session(cfg, stream.fps)
    session.reset(controller.RIGHT_READY_RAD, controller.LEFT_READY_RAD)

    q_right, q_left, q_torso, q_head = [], [], [], []
    q_right_hand, q_left_hand = [], []
    base_position, base_rotation = [], []
    eef_right, eef_left = [], []
    palm_right, palm_left = [], []
    wrist_right, wrist_left = [], []
    robot_torso = []
    human_left_sew, human_right_sew = [], []
    human_upper_rotation, human_upper_position = [], []
    left_mcp_centroid, right_mcp_centroid = [], []
    human_skeleton_positions = []
    solve_ms, contacts = [], []
    for index in range(count):
        frame = stream[index]
        human_left_sew.append(_optional_array(
            None if frame.left_sew is None else frame.left_sew.to_flat18(), (18,)
        ))
        human_right_sew.append(_optional_array(
            None if frame.right_sew is None else frame.right_sew.to_flat18(), (18,)
        ))
        human_upper_rotation.append(_optional_array(frame.R_world_upper_body, (3, 3)))
        human_upper_position.append(_optional_array(frame.p_world_upper_body, (3,)))
        left_mcp_centroid.append(_optional_array(
            frame.extras.get("left_finger_mcp_centroid"), (3,)
        ))
        right_mcp_centroid.append(_optional_array(
            frame.extras.get("right_finger_mcp_centroid"), (3,)
        ))
        if stream.skeleton_names is not None:
            human_skeleton_positions.append(_optional_array(
                None if frame.skeleton is None else frame.skeleton["positions"],
                (len(stream.skeleton_names), 3),
            ))
        started = time.perf_counter()
        out = session.solve(
            frame,
            q_current_right=controller.q_current_right,
            q_current_left=controller.q_current_left,
            q_current_right_hand=getattr(controller, "q_current_right_hand", None),
            q_current_left_hand=getattr(controller, "q_current_left_hand", None),
        )
        solve_ms.append(1e3 * (time.perf_counter() - started))
        _normalize_goals(out, controller)
        if out.p_world_base is not None and out.R_world_base is not None:
            controller.update_mocap_body(out.p_world_base, out.R_world_base)
        controller.set_joint_goals(out)
        controller.update_kinematic()

        q_right.append(np.asarray(out.q_goal_right, dtype=float))
        q_left.append(np.asarray(out.q_goal_left, dtype=float))
        torso = out.q_goal_torso
        if torso is None:
            torso = controller.q_current_torso
        q_torso.append(np.asarray(torso, dtype=float))
        head = out.q_goal_head if out.q_goal_head is not None else controller.q_current_head
        q_head.append(np.asarray(head, dtype=float))
        right_hand = out.q_goal_right_hand
        left_hand = out.q_goal_left_hand
        if right_hand is None:
            right_hand = np.zeros(12) if not q_right_hand else q_right_hand[-1]
        if left_hand is None:
            left_hand = np.zeros(12) if not q_left_hand else q_left_hand[-1]
        q_right_hand.append(np.asarray(right_hand, dtype=float))
        q_left_hand.append(np.asarray(left_hand, dtype=float))
        position = out.p_world_base
        rotation = out.R_world_base
        if position is None or rotation is None:
            position = np.zeros(3) if not base_position else base_position[-1]
            rotation = np.eye(3) if not base_rotation else base_rotation[-1]
        base_position.append(np.asarray(position, dtype=float))
        base_rotation.append(np.asarray(rotation, dtype=float))
        eef_right.append(_body_transform(model, data, "right_eef"))
        eef_left.append(_body_transform(model, data, "left_eef"))
        palm_right.append(_body_transform(model, data, "right_hand_link"))
        palm_left.append(_body_transform(model, data, "left_hand_link"))
        wrist_right.append(_body_transform(model, data, "link_right_arm_6"))
        wrist_left.append(_body_transform(model, data, "link_left_arm_6"))
        robot_torso.append(_body_transform(model, data, "link_torso_5"))
        contacts.append(_self_contacts(model, data))

    arrays = {
        "q_right": np.stack(q_right),
        "q_left": np.stack(q_left),
        "q_torso": np.stack(q_torso),
        "q_head": np.stack(q_head),
        "q_right_hand": np.stack(q_right_hand),
        "q_left_hand": np.stack(q_left_hand),
        "base_position": np.stack(base_position),
        "base_rotation": np.stack(base_rotation),
        "eef_right": np.stack(eef_right),
        "eef_left": np.stack(eef_left),
        "palm_right": np.stack(palm_right),
        "palm_left": np.stack(palm_left),
        "wrist_right": np.stack(wrist_right),
        "wrist_left": np.stack(wrist_left),
        "robot_torso": np.stack(robot_torso),
        "human_left_sew": np.stack(human_left_sew),
        "human_right_sew": np.stack(human_right_sew),
        "human_upper_rotation": np.stack(human_upper_rotation),
        "human_upper_position": np.stack(human_upper_position),
        "left_mcp_centroid": np.stack(left_mcp_centroid),
        "right_mcp_centroid": np.stack(right_mcp_centroid),
        "solve_ms": np.asarray(solve_ms),
        "self_contacts": np.asarray(contacts, dtype=np.int64),
    }
    if human_skeleton_positions:
        arrays["human_skeleton_positions"] = np.stack(human_skeleton_positions)
        arrays["human_skeleton_names"] = np.asarray(stream.skeleton_names).astype(str)
        arrays["human_skeleton_parents"] = np.asarray(
            stream.skeleton_parents, dtype=np.int64
        )
    arrays["base_delta"] = _base_deltas(
        arrays["base_position"], arrays["base_rotation"]
    )
    arrays["mocap_world_rotation"] = np.asarray(
        getattr(session, "R_mocap_world", np.eye(3)), dtype=float
    )
    arrays["mocap_world_position"] = np.asarray(
        getattr(session, "p_mocap_world", np.zeros(3)), dtype=float
    )
    for name, values in arrays.items():
        if name.startswith("q_") and not np.all(np.isfinite(values)):
            raise RuntimeError(f"{variant} produced non-finite {name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = output_dir / "trajectory.npz"
    np.savez_compressed(trajectory, **arrays)
    hdf5_path = write_episode(
        output_dir / "robot_data.hdf5" if hdf5_path is None else hdf5_path,
        arrays,
        demo_key=demo_key,
        fps=stream.fps,
        source=str(frames),
        variant=variant,
        config=asdict(cfg),
    )
    summary = {
        "variant": variant,
        "description": description,
        "solver": cfg.solver,
        "backend": f"{type(session).__module__}.{type(session).__name__}",
        "source_frames": str(frames),
        "frames": count,
        "fps": stream.fps,
        "solve_ms_mean": float(np.mean(arrays["solve_ms"])),
        "solve_ms_max": float(np.max(arrays["solve_ms"])),
        "self_collision_frames": int(np.count_nonzero(arrays["self_contacts"])),
        "trajectory": str(trajectory),
        "hdf5": str(hdf5_path),
        "demo_key": demo_key,
        "config": asdict(cfg),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=variant_names())
    parser.add_argument("--frames", default=str(SAMPLE_MOTION))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output = args.output_dir or f"outputs/offline_validation/{args.variant}"
    summary = run_validation(
        args.variant,
        frames=args.frames,
        output_dir=output,
        max_frames=args.max_frames,
        registry=args.registry,
    )
    print(json.dumps({k: summary[k] for k in (
        "variant", "backend", "frames", "solve_ms_mean",
        "solve_ms_max", "self_collision_frames",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
