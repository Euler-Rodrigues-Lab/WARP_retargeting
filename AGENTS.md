# Repository Contract

## Scope

This repository contains only the WARP paper code frozen at source checkpoint
`4f0fa5e` (2026-05-30), plus maintenance needed to package and reproduce that
same behavior.

Do not add post-WARP projects or experiments here. In particular, do not add
Aria/EgoPoser, perception pipelines, partial-mocap reconstruction, firm-grasp
post-processing, reviewer-response ablations, policy deployment, or later
hardware experiments.

## Protected implementation

Never copy analytic solver sources into this repository, including
`geometric_kinematics_*.py`, `constrained_sew.py`,
`geometric_subproblems*.py`, `sew_stereo*.py`, or legacy RBY1 SEW solver
modules. Consume the typed `geo_kin_core` interface through the pinned
`external/rby1_teleop` submodule.

## Ownership

- Repository and package author metadata must contain only `kczttm`.
- Robot models, controllers, SDK adapters, and hardware code belong upstream
  in `rby1_teleop`; update that repository first, then advance the submodule.
- Generated HDF5, CSV/CSV.GZ, video, image, and result data do not belong in
  Git. Small fixtures require an explicit allowlist entry.
- Every imported legacy file needs a source path, commit, and blob hash in
  `docs/PROVENANCE.md`.

Run `python scripts/check_repository_scope.py` and `pytest` before committing.
