# Provenance

New repository author: `kczttm`.

The original monorepo remains the archival history. This repository uses a
fresh, allowlisted import so protected solver blobs and unrelated projects do
not enter its Git object database.

| Destination | Source | Commit | Blob |
|---|---|---|---|
| `src/warp_retargeting/config.py` | `projects/xhand_teleop/retarget/configs/{schema,loader}.py` (packaged port) | `4f0fa5e` | `445e215a...`, `af4f5f07...` |
| `experiments/warp_paper/configs/warp_seed_no_joint_limits.yaml` | `scripts/warp_seed/example/run_config.yaml` | `80f92b6` | `5d4a1175...` |
| `experiments/warp_paper/configs/variants.yaml` | `scripts/hardware_ablation/variants.yaml` | `4f0fa5e` | `c32bc5570...` |
| RBY1 MINK fallback behavior in `geo_kin_core@79f6c4f` | `projects/xhand_teleop/mink_solver/{mink_setup,rby1_mink_solver}.py` (typed, protected-source-free port) | `4f0fa5e` | `6bb60ad6...`, `90ca275c...` |

Protected solver sources referenced by the paper are deliberately absent and
are consumed through the `rby1_teleop`/`geo_kin_core` interface.
