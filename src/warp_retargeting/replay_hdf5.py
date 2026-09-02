"""Replay frozen WARP robomimic HDF5 trajectories in MuJoCo."""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from geo_kin_core.types import RetargetFrame, RetargetOutput, SEWPose
from geo_kin_core.viz import HumanCapsuleViz, capsules
from rby1_teleop import XML_RBY1_XHAND
from rby1_teleop.control import RBY1WithXHandMuJoCoController

from .hdf5 import (
    ACT_BASE, ACT_HEAD, ACT_LEFT_ARM, ACT_LEFT_HAND, ACT_RIGHT_ARM,
    ACT_RIGHT_HAND, ACT_TORSO, OBS_HEAD, OBS_LEFT_ARM, OBS_RIGHT_ARM,
    OBS_TORSO, list_demos, load_episode,
)


class _Keys:
    def __init__(self, *, show_human: bool = False):
        self.paused = False
        self.restart = False
        self.step = False
        self.quit = False
        self.speed_scale = 1.0
        self.show_human = show_human

    def __call__(self, key):
        if key == ord(" "):
            self.paused = not self.paused
        elif key in (ord("r"), ord("R")):
            self.restart = True
        elif key == ord("."):
            self.step = True
        elif key == ord("]"):
            self.speed_scale *= 2.0
        elif key == ord("["):
            self.speed_scale *= 0.5
        elif key in (ord("q"), ord("Q")):
            self.quit = True
        elif key in (ord("h"), ord("H")):
            self.show_human = not self.show_human


def _human_source(episode) -> str | None:
    full = {
        "human_skeleton_positions", "human_skeleton_parents",
        "human_skeleton_names",
    }
    if full <= episode.keys():
        return f"captured {len(episode['human_skeleton_names'])}-bone skeleton"
    coarse = {
        "human_left_sew", "human_right_sew", "human_upper_rotation",
        "human_upper_position",
    }
    if coarse <= episode.keys():
        return "coarse SEW upper body"
    return None


def _human_frame(episode, index: int) -> RetargetFrame:
    """Rehydrate the stored targets for geo_kin_core's shared visualizer."""
    kwargs = {}
    if "human_skeleton_positions" in episode:
        kwargs["skeleton"] = {
            "positions": np.asarray(episode["human_skeleton_positions"])[index],
            "names": tuple(np.asarray(episode["human_skeleton_names"]).astype(str)),
            "parents": np.asarray(episode["human_skeleton_parents"], dtype=int),
        }
    else:
        for side in ("left", "right"):
            flat = np.asarray(episode[f"human_{side}_sew"])[index]
            if np.all(np.isfinite(flat)):
                kwargs[f"{side}_sew"] = SEWPose.from_flat18(flat)
        rotation = np.asarray(episode["human_upper_rotation"])[index]
        position = np.asarray(episode["human_upper_position"])[index]
        if np.all(np.isfinite(rotation)):
            kwargs["R_world_upper_body"] = rotation
        if np.all(np.isfinite(position)):
            kwargs["p_world_upper_body"] = position
    return RetargetFrame(**kwargs)


def _frame_output(episode, index: int, mode: str) -> RetargetOutput:
    if mode == "action":
        row = episode["actions_joint"][index]
        return RetargetOutput(
            q_goal_left=row[ACT_LEFT_ARM],
            q_goal_right=row[ACT_RIGHT_ARM],
            q_goal_torso=row[ACT_TORSO],
            q_goal_head=row[ACT_HEAD],
            q_goal_left_hand=row[ACT_LEFT_HAND],
            q_goal_right_hand=row[ACT_RIGHT_HAND],
        )
    row = episode["robot_pos"][index]
    return RetargetOutput(
        q_goal_left=row[OBS_LEFT_ARM],
        q_goal_right=row[OBS_RIGHT_ARM],
        q_goal_torso=row[OBS_TORSO],
        q_goal_head=row[OBS_HEAD],
        q_goal_left_hand=episode["q_left_hand"][index],
        q_goal_right_hand=episode["q_right_hand"][index],
    )


def replay(
    hdf5_file: str,
    *,
    demo_key: str = "demo_0",
    mode: str = "obs",
    speed: float = 1.0,
    loop: bool = False,
    headless: bool = False,
    max_frames: int | None = None,
    human_overlay: bool = False,
) -> int:
    episode = load_episode(hdf5_file, demo_key)
    model = mujoco.MjModel.from_xml_path(str(XML_RBY1_XHAND))
    data = mujoco.MjData(model)
    controller = RBY1WithXHandMuJoCoController(model, data, debug=False)
    controller.setup_mocap_body("base_mocap_mover")
    total = len(episode["robot_pos"])
    fps = float(episode["fps"])
    human_source = _human_source(episode)
    if human_overlay and human_source is None:
        raise ValueError(
            f"{hdf5_file}:{demo_key} contains no human targets; regenerate it "
            "with warp-validate-offline"
        )
    keys = _Keys(show_human=human_overlay)
    applied = 0

    if mode == "action":
        controller.set_joint_goals(_frame_output(episode, 0, "obs"))
        if "base_position" in episode:
            controller.update_mocap_body(
                episode["base_position"][0], episode["base_rotation"][0]
            )
        controller.update_kinematic()
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    def run(viewer=None, human_viz=None):
        nonlocal applied
        index = 0
        while viewer is None or viewer.is_running():
            started = time.perf_counter()
            if keys.quit:
                break
            if keys.restart:
                index = 0
                keys.restart = False
                mujoco.mj_resetData(model, data)
            advance = not keys.paused or keys.step
            keys.step = False
            if advance:
                output = _frame_output(episode, index, mode)
                controller.set_joint_goals(output)
                if "base_position" in episode:
                    controller.update_mocap_body(
                        episode["base_position"][index], episode["base_rotation"][index]
                    )
                # Both sources are applied kinematically, matching the policy-rollout
                # visualization path. ``obs`` selects recorded proprio while ``action``
                # selects recorded commands; neither pretends to be a dynamics benchmark.
                controller.update_kinematic()
                index += 1
                applied += 1
                if max_frames is not None and applied >= max_frames:
                    break
                if index >= total:
                    if loop:
                        index = 0
                    else:
                        break
            if viewer is not None:
                capsules.clear(viewer)
                if keys.show_human and human_viz is not None:
                    # ``index`` already points at the next frame after advancing.
                    drawn_index = (index - 1) % total if advance else index % total
                    human_viz.draw(_human_frame(episode, drawn_index))
                viewer.sync()
                delay = 1.0 / max(1e-6, fps * speed * keys.speed_scale)
                remaining = delay - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
        return applied

    print(
        f"Replaying {hdf5_file}:{demo_key} | {total} frames @ {fps:g} Hz | "
        f"variant={episode['variant'] or '<unknown>'} | mode={mode} | "
        f"human={human_source if human_overlay else 'off'}"
    )
    if headless:
        return run()
    from mujoco import viewer as mj_viewer
    with mj_viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False,
        key_callback=keys,
    ) as viewer:
        human_viz = HumanCapsuleViz(viewer)
        if "mocap_world_position" in episode and "mocap_world_rotation" in episode:
            human_viz.set_base_offset(
                episode["mocap_world_position"],
                np.asarray(episode["mocap_world_rotation"]).reshape(3, 3).T,
            )
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        print(
            "Controls: SPACE pause | R restart | H human overlay | "
            "[/] speed | . step | Q quit"
        )
        return run(viewer, human_viz)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5_file")
    parser.add_argument("--demo-key", default="demo_0")
    parser.add_argument(
        "--mode", choices=("obs", "action"), default="obs",
        help="Replay recorded proprio (obs) or recorded joint commands (action)",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--human-overlay-off", action="store_true",
        help="Overlay captured human skeleton capsules; press H to toggle",
    )
    parser.add_argument("--list-demos", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_demos:
        print("\n".join(list_demos(args.hdf5_file)))
        return 0
    replay(
        args.hdf5_file,
        demo_key=args.demo_key,
        mode=args.mode,
        speed=args.speed,
        loop=args.loop,
        headless=args.headless,
        max_frames=args.max_frames,
        human_overlay=not args.human_overlay_off,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
