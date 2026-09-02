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
| `fixtures/seed/washing_dishes_R_004__A299.csv.gz` | `scripts/warp_seed/example/input/washing_dishes_R_004__A299.csv.gz` | `80f92b6` | `8a3313f254632f841330e57fbbb426d53e4cc691` |
| `fixtures/seed/washing_dishes_R_004__A299.frames.npz` | Derived at 30 Hz from the preceding raw fixture by `warp_retargeting.transcode_csv` | local deterministic transcode | SHA-256 `21d70f3c55ac48e4280a34dffde512338bd619e9f17f99579e408ff4986b463d` |
| `src/warp_retargeting/hdf5.py` | Schema-compatible reimplementation of `projects/rby1_teleop/utils/data_collection.py` | `4f0fa5e` | `0cc6e4e6963d9aa8846036d088a2caf0c580f377` |
| `src/warp_retargeting/replay_hdf5.py` | Protected-source-free replay built from the original HDF5 loader/replayer behavior | `4f0fa5e` | `5aea6a2611d52f8388ae8bb5b320717c89bdeccd`, `7afa97c873ff38b8e6f3e69c6bd308d22db1d8a7` |
| `src/warp_retargeting/policy_rollout.py` | Protected-source-free compatibility implementation of the RBY1 simulation rollout protocol | `4f0fa5e` | `f96e1b8715bb4a074cc3538162fe9d860256bf57` |
| `src/warp_retargeting/metrics.py` | Independent portable subset using saved targets; vocabulary/layout follows the frozen metric runner | `4f0fa5e` | `7a8831e991ff3db01174e2eef01e0437f0d2e204`, `b833ac2aa0e9b7f581fc482c284be5dc478b2715` |
| Licensed mobile-base variant configuration | `geo_kin` PyO3 session boundary | `9dbd4ed` | Explicit paper knobs; no solver source enters this repository |

Protected solver sources referenced by the paper are deliberately absent and
are consumed through the `rby1_teleop`/`geo_kin_core` interface.

The raw canonical fixture SHA-256 is
`c2e18c60407a3362e659efec1421d587de25a652c3d00128020fe4db78907578`.
The eight additional user-supplied recordings are external validation inputs
only and are not repository artifacts.
