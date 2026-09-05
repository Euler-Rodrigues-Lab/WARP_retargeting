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

For an existing checkout, initialize both pinned public dependencies before
syncing:

```bash
git pull
git submodule update --init --recursive
uv sync --extra runtime --extra test
```

The sync installs `external/rby1_teleop`, `external/geo_kin_core`, HDF5
support, the public MINK fallback, and the `geo-kin-provision` command.
Register a supplied licensed wheel and license once per user:

```bash
uv run geo-kin-provision register \
  --product rby1-xhand \
  --wheel /path/to/geo_kin-0.1.0-cp310-abi3-manylinux_2_35_x86_64.whl \
  --license /path/to/geo_kin_license.toml \
  --name my-rby1-license \
  --activate
```

Then link that central build into this environment when WARP/C-SEW parity is
required:

```bash
uv run geo-kin-provision install
```

The wheel is unpacked once under `~/.local/share/geo-kin`; licenses live under
`~/.config/geo-kin`. Private artifacts are never stored in this repository or
its submodules.

## Offline validation

Choose a frozen variant. `warp`, `warp_no_joint_limits`, `warp_no_spring`,
`c_sew`, and `sew_mimic` require the licensed wheel; MINK runs through the
public `geo_kin_core` fallback.

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
  --variant warp \
  --frames fixtures/seed/washing_dishes_R_004__A299.frames.npz \
  --output-dir outputs/warp

uv run warp-replay-hdf5 outputs/warp/robot_data.hdf5 --loop
uv run warp-metrics outputs/warp/robot_data.hdf5
uv run warp-rollout-policy \
  --dataset outputs/warp/robot_data.hdf5 \
  --use-gt-action
```

Replay is deliberately kinematic, matching the frozen policy-visualization
script; it does not claim to evaluate MuJoCo dynamics. Use `--mode action` to
select recorded commands rather than recorded proprio, `--headless` for CI,
and `--list-demos` to inspect a multi-demo file. The captured 70-bone SEED pose
is drawn as translucent capsules by default; press `H` while replaying to
toggle it or pass `--no-human-overlay` to start without it. Older generated
HDF5 files fall back to a coarse upper-body overlay from their stored SEW
targets. Regenerate them to preserve the complete captured body and finger
skeleton. Replay delegates both forms to the shared
`geo_kin_core.viz.HumanCapsuleViz` implementation, which was ported from
`SEW-Geometric-Teleop`; WARP does not define a second human model.

## License-free canonical replays

The repository includes precomputed HDF5 trajectories for every frozen mode
that normally requires the licensed `geo_kin` backend. These are derived robot
joint targets and replay metadata only; they contain no solver implementation
or license material. A fresh clone can replay them with the public runtime:

```bash
uv sync --extra runtime

uv run warp-replay-hdf5 \
  fixtures/retargeted/washing_dishes_warp.hdf5 --loop
uv run warp-replay-hdf5 \
  fixtures/retargeted/washing_dishes_c_sew.hdf5 --loop
```

Replace the filename suffix with `warp_no_spring`, `warp_no_joint_limits`, or
`sew_mimic` to compare another licensed configuration. The licensed wheel is
required only to retarget a new frame stream or regenerate these frozen
outputs, not to replay them.

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
uv run geo-kin-provision install

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
