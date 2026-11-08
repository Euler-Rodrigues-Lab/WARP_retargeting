# WARP

### Whole-Body Retargeting for Learning from Offline Human Demonstrations

**CoRL 2026 · Georgia Institute of Technology**

[Project website](https://warp-retargeting.github.io/) ·
[Paper](https://arxiv.org/abs/2606.29940) ·
[Installation and reproduction guide](docs/RUNNING.md)

WARP turns offline human motion recordings into whole-body robot actions for
mobile manipulation. Its geometric retargeter preserves the intent of human
motion while aligning the robot's hands, coordinating the torso, and moving
the base when needed. The resulting trajectories support open-loop replay and
policy learning without collecting robot teleoperation demonstrations.

The method combines a closed-form Shoulder–Elbow–Wrist solver (c-SEW), adaptive
body alignment, and lazy mobile-base control. Visit the project website for
videos, method illustrations, and simulation and real-world results.

## What's included

- RB-Y1 with XHand replay examples, including a human-motion overlay.
- WARP, c-SEW, SEW-mimic, and MINK retargeting configurations.
- SEED recording conversion, HDF5 dataset export, and trajectory metrics.
- A policy rollout client for the paper's websocket interface.

This release provides the retargeting and replay workflows. Policy training
and the full paper evaluation pipeline are separate; see
[workflow coverage and reproduction limits](docs/WORKFLOWS.md).

## Installation

Use Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/)
0.13.7 or later. Interactive replay also needs a graphical desktop with OpenGL.
If uv is already installed, check `uv --version` and upgrade it with
`uv self update` (or `python -m pip install --upgrade uv` for a pip installation).

```bash
git clone --recurse-submodules https://github.com/Euler-Rodrigues-Lab/WARP_retargeting.git
cd WARP_retargeting
uv sync --locked --extra runtime
```

Core and recording support come from the published
[`eurola-geo-kin-core`](https://pypi.org/project/eurola-geo-kin-core/0.1.2/) and
[`eurola-xrt-devices`](https://pypi.org/project/eurola-xrt-devices/0.1.2/)
packages, pinned to 0.1.2. The RB-Y1 model and integration remain a pinned
submodule. For an existing checkout, run `git submodule update --init --recursive`
before syncing.

## Try the demo

Replay the included washing-dishes trajectory—no solver license or robot needed:

```bash
uv run warp-replay-hdf5 fixtures/retargeted/washing_dishes_warp.hdf5 --loop
```

Press **H** to toggle the human skeleton overlay. Close the viewer to exit.
To verify playback without opening a window:

```bash
uv run warp-replay-hdf5 fixtures/retargeted/washing_dishes_warp.hdf5 \
  --headless --max-frames 120
```

Replay displays recorded robot configurations; it is kinematic playback.
Other included trajectories are listed in [the fixture guide](fixtures/retargeted/README.md).

## Retarget a recording

Start with the public MINK baseline on the included human-motion clip:

```bash
uv run warp-validate-offline \
  --variant mink_eef \
  --frames fixtures/seed/washing_dishes_R_004__A299.frames.npz \
  --max-frames 120 \
  --output-dir outputs/mink_eef

uv run warp-replay-hdf5 outputs/mink_eef/robot_data.hdf5 --loop
uv run warp-metrics outputs/mink_eef/robot_data.hdf5
```

The output directory contains `robot_data.hdf5`, `trajectory.npz`, and
`summary.json`. MINK retargets the arms and body; analytical hand retargeting
is not included in this public baseline.

To generate new WARP trajectories, install the separately supplied licensed
`geo_kin` solver using the [solver setup guide](docs/RUNNING.md#installation),
then replace `mink_eef` with `warp` above. Replaying the precomputed WARP
trajectories does **not** require that solver.

For your own SEED recordings, dataset creation, and policy rollout, follow the
[full run guide](docs/RUNNING.md).

## Development

```bash
uv sync --locked --extra runtime --extra test
uv run python scripts/check_repository_scope.py
uv run pytest
```

See [scope](docs/SCOPE.md) and [source provenance](docs/PROVENANCE.md) for the
paper implementation boundary.

## Citation

Citation from the [project website](https://warp-retargeting.github.io/):

```bibtex
@inproceedings{chen2026warp,
  title = {{WARP}: Whole-Body Retargeting for Learning from Offline Human Demonstrations},
  author = {Chen, Zhenyang and Kong, Chuizheng and Zhang, Chuye and Yang, Yuanshao and Zhu, Lawrence Y. and Kousik, Shreyas and Xu, Danfei},
  booktitle = {Proceedings of the 10th Conference on Robot Learning (CoRL)},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  year = {2026},
  eprint = {2606.29940},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2606.29940}
}
```

## License

Repository code is covered by the [MIT license](LICENSE). Dependencies retain
their own licenses; the analytical `geo_kin` backend is licensed separately.
