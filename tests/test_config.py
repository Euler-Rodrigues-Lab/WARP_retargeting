from pathlib import Path

import pytest

from warp_retargeting.config import ConfigError, dump_run_config, load_config


ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG = ROOT / "experiments/warp_paper/configs/warp_seed_no_joint_limits.yaml"


def test_frozen_paper_config_invariants():
    cfg = load_config(PAPER_CONFIG)
    assert cfg.solver == "sew"
    assert cfg.data.no_filter is True
    assert cfg.enable_base_motion is True
    assert cfg.sew is not None
    assert cfg.sew.enable_functional_retargeting is True
    assert cfg.sew.enable_functional_offset is True
    assert cfg.sew.enable_joint_limits is False
    assert cfg.sew.enable_collision_avoidance is False
    assert cfg.sew.mobile_base_spring_damper is False


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("solver: sew\nio:\n  input_folder: .\nsew:\n  enable_joint_limit: true\n")
    with pytest.raises(ConfigError, match="did you mean 'enable_joint_limits'"):
        load_config(path)


def test_dump_omits_inactive_solver(tmp_path):
    cfg = load_config(PAPER_CONFIG)
    output = tmp_path / "effective.yaml"
    dump_run_config(cfg, output)
    text = output.read_text()
    assert "sew:" in text
    assert "mink:" not in text
