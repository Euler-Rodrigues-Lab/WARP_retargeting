# WARP Retargeting

Reproducibility and experiment tooling for **WARP** (Whole-Body Retargeting
for Learning from Offline Human Demonstrations).

**[Project website](https://warp-retargeting.github.io/)** ·
**[Paper](https://arxiv.org/abs/2606.29940)**

This repository is intentionally limited to the code path frozen for the WARP
submission on 2026-05-30. Work started after that boundary—including
Aria/EgoPoser perception, partial-mocap reconstruction, reviewer-response
experiments, reviewer-era policy deployment changes, and later hardware
development—belongs in separate repositories. The frozen downstream policy
validation path is included.

The RB-Y1 robot integration is pinned as a submodule at
`external/rby1_teleop`. Protected closed-form kinematics are not stored here;
they are accessed through the `geo_kin_core` session API and the separately
licensed `geo_kin` backend used by `rby1_teleop`.

## Bootstrap

```bash
git clone --recurse-submodules https://github.com/Euler-Rodrigues-Lab/WARP_retargeting.git
cd WARP_retargeting
uv sync --extra runtime --extra test
python scripts/check_repository_scope.py
uv run pytest
```

This installs the pinned `external/rby1_teleop` checkout, HDF5 support, and the
public MINK fallback. Install the licensed `geo_kin` wheel built from
`geo_kin@9dbd4ed` or newer after syncing when WARP/C-SEW parity is required:

```bash
uv pip install --python .venv/bin/python /path/to/geo_kin.whl
```

## Offline validation

Choose a frozen variant. `warp_seed`, `ours`, `c_sew`, and `sew_mimic` require
the licensed wheel; MINK runs through the public `geo_kin_core` fallback.

```bash
uv run warp-validate-offline --variant mink_eef --max-frames 120
uv run warp-validate-offline --variant mink_te --max-frames 120
uv run warp-validate-offline --variant c_sew --max-frames 120
uv run warp-validate-offline --variant sew_mimic --max-frames 120
```

Each run writes `summary.json`, a solver-neutral `trajectory.npz`, and the
original robomimic-style `robot_data.hdf5`. MINK hand outputs remain unsolved:
the paper baseline reused protected analytical XHand IK, which is deliberately
not copied into this repository.

## Canonical paper replay

The allowlisted fixture contains the original SEED bone CSV and a derived 420
frame, 30 Hz cache. This reproduces the submitted WARP converter settings:

```bash
uv run warp-validate-offline \
  --variant warp_seed \
  --frames fixtures/seed/washing_dishes_R_004__A299.frames.npz \
  --output-dir outputs/warp_seed

uv run warp-replay-hdf5 outputs/warp_seed/robot_data.hdf5 --loop
uv run warp-metrics outputs/warp_seed/robot_data.hdf5
uv run warp-rollout-policy \
  --dataset outputs/warp_seed/robot_data.hdf5 \
  --use-gt-action
```

Replay is deliberately kinematic, matching the frozen policy-visualization
script; it does not claim to evaluate MuJoCo dynamics. Use `--mode action` to
select recorded commands rather than recorded proprio, `--headless` for CI,
and `--list-demos` to inspect a multi-demo file.

## More SEED recordings

Raw `.csv`/`.csv.gz` recordings first become solver-neutral frame streams. The
capture-side reader is temporarily sourced from a frozen
`SEW-Geometric-Teleop` checkout, so run this command in an environment that has
that checkout's device dependencies:

```bash
python -m warp_retargeting.transcode_csv \
  --csv-file /path/to/recording.csv.gz \
  --output /path/to/recording.frames.npz \
  --monolith-path /path/to/SEW-Geometric-Teleop
```

Build any number of frame streams into one original-format dataset:

```bash
uv run warp-build-dataset \
  --variant c_sew \
  --output-dir outputs/c_sew_samples \
  /path/to/frames/*.npz

uv run warp-metrics \
  outputs/c_sew_samples/robot_data.hdf5 \
  --all-demos
```

The eight locally supplied recordings were validated through this path without
being copied into Git; see [docs/SAMPLE_VALIDATION.md](docs/SAMPLE_VALIDATION.md).

## Policy rollout

Ground-truth action rollout needs only the runtime extra. For the frozen
msgpack/websocket policy interface, add the policy dependencies:

```bash
uv sync --extra runtime --extra policy
uv pip install --python .venv/bin/python /path/to/geo_kin.whl

uv run warp-rollout-policy \
  --dataset /path/to/robot_data.hdf5 \
  --demo-key demo_0 \
  --host 127.0.0.1 --port 8000
```

The client accepts the frozen 14D, 25D, 44D, and WARP 49D action layouts. Use
`--action-layout base_first_49` for the alternate base-first 49D checkpoint.
It honors policy `camera_keys`/`proprio_keys`, supports dataset observations
with `--use-gt-observation`, and can render a MuJoCo view or send zero images.
The portable metric command currently covers end-effector position and
orientation, torso orientation, joint velocity/jerk, and self-collision. It is
not a replacement for the separate paper aggregation/figure repository or its
global cross-demo NNAD/PCAV pass.

The exact compatibility boundary is recorded in
[docs/WORKFLOWS.md](docs/WORKFLOWS.md).

## Citation

If you use WARP, please cite the paper using the format published on the
[project website](https://warp-retargeting.github.io/):

```bibtex
@article{chen2026warp,
  title   = {WARP: Whole-Body Retargeting for Learning from Offline Human Demonstrations},
  author  = {Chen, Zhenyang and Kong, Chuizheng and Zhang, Chuye and Yang, Yuanshao and Zhu, Lawrence Y. and Kousik, Shreyas and Xu, Danfei},
  journal = {arXiv preprint arXiv:2606.29940},
  year    = {2026}
}
```
