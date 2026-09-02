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

Protected solver sources referenced by the paper are deliberately absent and
are consumed through the `rby1_teleop`/`geo_kin_core` interface.
