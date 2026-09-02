from warp_retargeting.variants import load_variant, variant_names


def test_expected_frozen_variants_are_registered():
    assert {"ours", "sew_mimic", "mink_eef", "mink_te"} <= set(variant_names())


def test_mink_eef_materializes_frozen_costs(tmp_path):
    cfg, description = load_variant(
        "mink_eef",
        input_folder=str(tmp_path),
        output_folder=str(tmp_path / "out"),
    )
    assert "MINK-EF" in description
    assert cfg.solver == "mink"
    assert cfg.mink.hand_position_cost == 1.0
    assert cfg.mink.torso_orientation_cost == 0.0
    assert cfg.mink.elbow_angle_cost == 0.0
    assert cfg.mink.wrist_target_mode == "palm"


def test_sew_mimic_disables_functional_retargeting(tmp_path):
    cfg, _ = load_variant("sew_mimic", input_folder=str(tmp_path))
    assert cfg.solver == "sew"
    assert cfg.sew.enable_functional_retargeting is False
    assert cfg.sew.enable_functional_offset is False
    assert cfg.sew.mobile_base_spring_damper is False
