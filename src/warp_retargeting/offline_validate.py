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
    )


def run_validation(
    variant: str,
    *,
    frames: str | Path = SAMPLE_MOTION,
    output_dir: str | Path,
    max_frames: int | None = None,
    registry: str | Path = DEFAULT_REGISTRY,
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
    solve_ms, contacts = [], []
    for index in range(count):
        frame = stream[index]
        started = time.perf_counter()
        out = session.solve(
            frame,
            q_current_right=controller.q_current_right,
            q_current_left=controller.q_current_left,
            q_current_right_hand=getattr(controller, "q_current_right_hand", None),
            q_current_left_hand=getattr(controller, "q_current_left_hand", None),
        )
        solve_ms.append(1e3 * (time.perf_counter() - started))
        if out.p_world_base is not None and out.R_world_base is not None:
            controller.update_mocap_body(out.p_world_base, out.R_world_base)
        controller.set_joint_goals(out)
        controller.update_kinematic()

        q_right.append(np.asarray(out.q_goal_right, dtype=float))
        q_left.append(np.asarray(out.q_goal_left, dtype=float))
        q_torso.append(np.asarray(out.q_goal_torso, dtype=float))
        head = out.q_goal_head if out.q_goal_head is not None else controller.q_current_head
        q_head.append(np.asarray(head, dtype=float))
        contacts.append(_self_contacts(model, data))

    arrays = {
        "q_right": np.stack(q_right),
        "q_left": np.stack(q_left),
        "q_torso": np.stack(q_torso),
        "q_head": np.stack(q_head),
        "solve_ms": np.asarray(solve_ms),
        "self_contacts": np.asarray(contacts, dtype=np.int64),
    }
    for name, values in arrays.items():
        if name.startswith("q_") and not np.all(np.isfinite(values)):
            raise RuntimeError(f"{variant} produced non-finite {name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = output_dir / "trajectory.npz"
    np.savez_compressed(trajectory, **arrays)
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
