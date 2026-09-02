import h5py
import numpy as np

from geo_kin_core.types import RetargetOutput

from warp_retargeting.hdf5 import load_episode, write_episode
from warp_retargeting.policy_rollout import map_action


def _arrays(n=3):
    return {
        "q_left": np.zeros((n, 7)),
        "q_right": np.ones((n, 7)),
        "q_torso": np.zeros((n, 6)),
        "q_head": np.zeros((n, 2)),
        "q_left_hand": np.zeros((n, 12)),
        "q_right_hand": np.ones((n, 12)),
        "base_delta": np.zeros((n, 3)),
        "base_position": np.zeros((n, 3)),
        "base_rotation": np.repeat(np.eye(3)[None], n, axis=0),
        "eef_left": np.zeros((n, 16)),
        "eef_right": np.zeros((n, 16)),
        "solve_ms": np.ones(n),
        "self_contacts": np.zeros(n),
    }


def test_original_robomimic_layout_round_trips(tmp_path):
    path = write_episode(tmp_path / "robot_data.hdf5", _arrays(), variant="c_sew")
    with h5py.File(path, "r") as root:
        demo = root["data/demo_0"]
        assert demo["actions/joint"].shape == (3, 49)
        assert demo["actions/joint_arm_head_torso_base"].shape == (3, 25)
        assert demo["obs/robot0_joint_pos"].shape == (3, 26)
        assert demo["obs/cmd_ts"].shape == (3,)
        assert demo.attrs["variant"] == "c_sew"
    loaded = load_episode(path)
    assert loaded["actions_joint"].shape == (3, 49)


def test_policy_action_layouts_preserve_unmapped_groups():
    held = RetargetOutput(
        q_goal_left=np.zeros(7), q_goal_right=np.zeros(7),
        q_goal_torso=np.zeros(6), q_goal_head=np.zeros(2),
        q_goal_left_hand=np.zeros(12), q_goal_right_hand=np.zeros(12),
    )
    output, base = map_action(np.arange(49), held)
    np.testing.assert_array_equal(output.q_goal_left, np.arange(7))
    np.testing.assert_array_equal(output.q_goal_right, np.arange(7, 14))
    np.testing.assert_array_equal(base, np.arange(22, 25))

    output, base = map_action(np.arange(14), held)
    np.testing.assert_array_equal(output.q_goal_torso, np.zeros(6))
    np.testing.assert_array_equal(base, np.zeros(3))

    output, base = map_action(np.arange(49), held, "base_first_49")
    np.testing.assert_array_equal(base, np.arange(3))
    np.testing.assert_array_equal(output.q_goal_torso, np.arange(3, 9))
    np.testing.assert_array_equal(output.q_goal_left, np.arange(11, 18))
