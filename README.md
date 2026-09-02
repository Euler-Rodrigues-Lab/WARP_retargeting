# WARP Retargeting

Reproducibility and experiment tooling for **WARP** (Whole-Body Retargeting
for Learning from Offline Human Demonstrations).

This repository is intentionally limited to the code path frozen for the WARP
submission on 2026-05-30. Work started after that boundary—including
Aria/EgoPoser perception, partial-mocap reconstruction, reviewer-response
experiments, policy deployment, and later hardware development—belongs in
separate repositories.

The RB-Y1 robot integration is pinned as a submodule at
`external/rby1_teleop`. Protected closed-form kinematics are not stored here;
they are accessed through the `geo_kin_core` session API and the separately
licensed `geo_kin` backend used by `rby1_teleop`.

## Bootstrap

```bash
git clone --recurse-submodules https://github.com/Euler-Rodrigues-Lab/WARP_retargeting.git
cd WARP_retargeting
python -m pip install -e '.[test]'
python scripts/check_repository_scope.py
pytest
```

For solver/runtime work, use `uv sync --extra runtime`; this installs the
pinned `external/rby1_teleop` checkout and resolves `geo-kin-core`. Install the
licensed `geo_kin` wheel separately when licensed-backend parity is required.

The current milestone provides the frozen configuration surface and repository
guardrails. The legacy CSV-to-HDF5 driver will be ported behind the
`rby1_teleop` API without copying solver or robot implementation files.

## Author

`kczttm`
