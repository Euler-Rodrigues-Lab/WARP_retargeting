"""RBY1 MuJoCo rollout from frozen HDF5 actions or a websocket policy."""

from __future__ import annotations

import argparse
from collections import deque
import time

import h5py
import mujoco
import numpy as np

from geo_kin_core.types import RetargetOutput
from rby1_teleop import XML_RBY1_XHAND
from rby1_teleop.control import RBY1WithXHandMuJoCoController

from .hdf5 import (
    ACT_BASE, ACT_HEAD, ACT_LEFT_ARM, ACT_LEFT_HAND, ACT_RIGHT_ARM,
    ACT_RIGHT_HAND, ACT_TORSO, OBS_HEAD, OBS_LEFT_ARM, OBS_RIGHT_ARM, OBS_TORSO,
)


def normalize_action_chunk(chunk) -> np.ndarray:
    values = np.asarray(chunk, dtype=np.float64)
    if values.ndim == 3:
        values = values[0]
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError(f"policy action chunk must be (T,D) or (1,T,D), got {values.shape}")
    return values


def map_action(
    row, held: RetargetOutput, layout: str = "auto"
) -> tuple[RetargetOutput, np.ndarray]:
    """Map the frozen policy action layouts to one typed RBY1 command."""
    row = np.asarray(row, dtype=float).reshape(-1)
    output = RetargetOutput(
        q_goal_left=np.asarray(held.q_goal_left).copy(),
        q_goal_right=np.asarray(held.q_goal_right).copy(),
        q_goal_torso=np.asarray(held.q_goal_torso).copy(),
        q_goal_head=np.asarray(held.q_goal_head).copy(),
        q_goal_left_hand=np.asarray(held.q_goal_left_hand).copy(),
        q_goal_right_hand=np.asarray(held.q_goal_right_hand).copy(),
    )
    base_delta = np.zeros(3)
    if len(row) == 49 and layout == "base_first_49":
        base_delta = row[0:3]
        output.q_goal_torso = row[3:9]
        output.q_goal_head = row[9:11]
        output.q_goal_left = row[11:18]
        output.q_goal_right = row[18:25]
        output.q_goal_left_hand = row[25:37]
        output.q_goal_right_hand = row[37:49]
    elif len(row) == 49:
        output.q_goal_left = row[ACT_LEFT_ARM]
        output.q_goal_right = row[ACT_RIGHT_ARM]
        output.q_goal_torso = row[ACT_TORSO]
        output.q_goal_head = row[ACT_HEAD]
        output.q_goal_left_hand = row[ACT_LEFT_HAND]
        output.q_goal_right_hand = row[ACT_RIGHT_HAND]
        base_delta = row[ACT_BASE]
    elif len(row) == 44:  # arms + hands + torso
        output.q_goal_left = row[0:7]
        output.q_goal_right = row[7:14]
        output.q_goal_left_hand = row[14:26]
        output.q_goal_right_hand = row[26:38]
        output.q_goal_torso = row[38:44]
    elif len(row) == 25:  # arms + torso + head + SE(2) base
        output.q_goal_left = row[0:7]
        output.q_goal_right = row[7:14]
        output.q_goal_torso = row[14:20]
        output.q_goal_head = row[20:22]
        base_delta = row[22:25]
    elif len(row) == 14:
        output.q_goal_left = row[0:7]
        output.q_goal_right = row[7:14]
    else:
        raise ValueError(
            f"unsupported policy action dimension {len(row)}; expected 14, 25, 44, or 49"
        )
    return output, np.asarray(base_delta, dtype=float)


class HDF5PolicyDataset:
    def __init__(self, path: str, demo_key: str, action_key: str):
        self.root = h5py.File(path, "r")
        self.demo = self.root["data"][demo_key]
        self.obs = self.demo["obs"]
        self.actions = self.demo["actions"][action_key]
        self.n = int(self.demo.attrs["num_samples"])

    def close(self):
        self.root.close()

    def observation(self, index: int) -> dict[str, np.ndarray]:
        return {key: value[index] for key, value in self.obs.items() if isinstance(value, h5py.Dataset)}

    def action(self, index: int) -> np.ndarray:
        return np.asarray(self.actions[index])


class WebPolicyClient:
    def __init__(self, host: str, port: int):
        try:
            import msgpack_numpy
            import websockets.sync.client
        except ImportError as exc:
            raise RuntimeError(
                "websocket rollout needs the policy extra: `uv sync --extra policy`"
            ) from exc
        msgpack_numpy.patch()
        self.codec = msgpack_numpy
        self.connection = websockets.sync.client.connect(
            f"ws://{host}:{port}", compression=None, max_size=None
        )
        self.metadata = msgpack_numpy.unpackb(self.connection.recv())

    def infer(self, observation: dict) -> np.ndarray:
        self.connection.send(self.codec.packb(observation))
        response = self.connection.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        unpacked = self.codec.unpackb(response)
        if isinstance(unpacked, dict):
            for key in ("actions", "action", "action_chunk"):
                if key in unpacked:
                    return normalize_action_chunk(unpacked[key])
        return normalize_action_chunk(unpacked)

    def close(self):
        self.connection.close()


def _initial_output(row, left_hand, right_hand) -> RetargetOutput:
    return RetargetOutput(
        q_goal_left=row[OBS_LEFT_ARM], q_goal_right=row[OBS_RIGHT_ARM],
        q_goal_torso=row[OBS_TORSO], q_goal_head=row[OBS_HEAD],
        q_goal_left_hand=left_hand, q_goal_right_hand=right_hand,
    )


def run(args) -> int:
    dataset = HDF5PolicyDataset(args.dataset, args.demo_key, args.action_key)
    client = None if args.use_gt_action else WebPolicyClient(args.host, args.port)
    model = mujoco.MjModel.from_xml_path(str(XML_RBY1_XHAND))
    data = mujoco.MjData(model)
    controller = RBY1WithXHandMuJoCoController(model, data, debug=False)
    controller.setup_mocap_body("base_mocap_mover")
    first = dataset.observation(min(args.initial_step, dataset.n - 1))
    held = _initial_output(
        first["robot0_joint_pos"], first.get("hand_left_qpos", np.zeros(12)),
        first.get("hand_right_qpos", np.zeros(12)),
    )
    controller.set_joint_goals(held)
    controller.update_kinematic()
    queue: deque[np.ndarray] = deque()
    base_pose = np.zeros(3)
    steps = 0
    metadata = {} if client is None else client.metadata
    camera_keys = list(metadata.get("camera_keys") or [])
    proprio_keys = list(metadata.get("proprio_keys") or ["robot0_joint_pos"])
    renderer = None
    if client is not None and camera_keys and not args.use_gt_observation and not args.zero_obs:
        renderer = mujoco.Renderer(model, height=args.image_height, width=args.image_width)

    def live_observation(index: int) -> dict:
        if args.use_gt_observation:
            observation = dataset.observation(index % dataset.n)
            requested = list(dict.fromkeys(camera_keys + proprio_keys))
            missing = [key for key in requested if key not in observation]
            if missing:
                raise KeyError(
                    f"dataset observation is missing policy keys: {', '.join(missing)}"
                )
            payload = {key: observation[key] for key in requested}
            payload["robot0_joint_pos_no_wheel"] = np.asarray(
                observation["robot0_joint_pos"][4:], dtype=np.float32
            )
            return payload
        robot = np.concatenate([
            np.zeros(4), controller.q_current_torso, controller.q_current_right,
            controller.q_current_left, controller.q_current_head,
        ]).astype(np.float32)
        proprio = {
            "robot0_joint_pos": robot,
            "robot0_joint_pos_no_wheel": robot[4:],
            "right_arm": np.asarray(controller.q_current_right, dtype=np.float32),
            "left_arm": np.asarray(controller.q_current_left, dtype=np.float32),
            "hand_right_qpos": np.asarray(controller.q_current_right_hand, dtype=np.float32),
            "hand_left_qpos": np.asarray(controller.q_current_left_hand, dtype=np.float32),
        }
        payload = {}
        for key in proprio_keys:
            if key not in proprio:
                raise KeyError(f"unsupported live MuJoCo proprio key: {key}")
            payload[key] = proprio[key]
        if camera_keys:
            if args.zero_obs:
                image = np.zeros((args.image_height, args.image_width, 3), dtype=np.uint8)
            else:
                assert renderer is not None
                renderer.update_scene(data, camera=args.camera_id)
                image = np.asarray(renderer.render(), dtype=np.uint8)[::-1, :, ::-1]
            for key in camera_keys:
                payload[key] = image.copy()
        if "task_id" in proprio_keys or metadata.get("task_id") is not None:
            task = np.zeros(64, dtype=np.float32)
            task[args.task_id] = 1.0
            payload["task_id"] = task
        return payload

    def loop(viewer=None):
        nonlocal held, base_pose, steps
        while viewer is None or viewer.is_running():
            started = time.perf_counter()
            index = steps % dataset.n
            if args.use_gt_action:
                action = dataset.action(index)
            else:
                if not queue:
                    queue.extend(client.infer(live_observation(index)))
                action = queue.popleft()
            held, base_delta = map_action(action, held, args.action_layout)
            base_pose += base_delta
            yaw = base_pose[2]
            rotation = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ])
            controller.update_mocap_body(np.array([base_pose[0], base_pose[1], 0.0]), rotation)
            controller.set_joint_goals(held)
            controller.update_kinematic()
            steps += 1
            if viewer is not None:
                viewer.sync()
            if args.max_steps is not None and steps >= args.max_steps:
                break
            delay = 1.0 / args.frequency - (time.perf_counter() - started)
            if viewer is not None and delay > 0:
                time.sleep(delay)

    try:
        if args.headless:
            loop()
        else:
            from mujoco import viewer as mj_viewer
            with mj_viewer.launch_passive(
                model=model, data=data, show_left_ui=False, show_right_ui=False,
            ) as viewer:
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 180
                viewer.cam.elevation = -15
                viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
                loop(viewer)
    finally:
        dataset.close()
        if client is not None:
            client.close()
        if renderer is not None:
            renderer.close()
    print(f"Completed {steps} rollout steps")
    return steps


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--demo-key", default="demo_0")
    parser.add_argument("--action-key", default="joint")
    parser.add_argument(
        "--action-layout", choices=("auto", "warp_49", "base_first_49"),
        default="auto", help="Disambiguate the two frozen 49D policy layouts",
    )
    parser.add_argument("--use-gt-action", action="store_true")
    parser.add_argument("--use-gt-observation", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--frequency", type=float, default=10.0)
    parser.add_argument("--task-id", type=int, default=0, choices=range(64))
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--camera-id", type=int, default=-1)
    parser.add_argument("--zero-obs", action="store_true")
    parser.add_argument("--initial-step", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
