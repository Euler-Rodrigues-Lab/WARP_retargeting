# Frozen workflow compatibility

The target is the RBY1 portion of
`SEW-Geometric-Teleop@4f0fa5e` on
`proj_func_retarget/baseline-metric-study`. The later curated commit
`80f92b6` is a map of that work, not a scope expansion.

| Frozen operation | WARP Retargeting entry point | Status |
|---|---|---|
| SEED bone CSV sampling at 30 Hz | `warp-transcode-csv` | Compatible; temporarily needs the frozen monolith's capture/device environment |
| WARP, C-SEW, SEW-mimic conversion | `warp-validate-offline`, `warp-build-dataset` | Licensed `geo_kin` backend; protected solver source is absent |
| MINK-EF / MINK-TE conversion | same commands with `mink_eef` / `mink_te` | Public `geo_kin_core` fallback; arm/body path complete, protected XHand IK intentionally absent |
| Robomimic HDF5 output | `robot_data.hdf5` from either converter | Original 49D/38D/25D/14D action groups and 26D robot state layout; policy rollout also accepts the frozen 44D arms/hands/torso layout |
| Multi-demo conversion | `warp-build-dataset` | One `data/demo_N` group per input, plus per-demo summaries and caches |
| MuJoCo data replay | `warp-replay-hdf5` | Recorded proprio or command replay; interactive and headless; kinematic by design |
| Metric execution | `warp-metrics` | Portable local subset: EEF position/orientation, torso orientation, joint velocity/jerk, self-collision |
| Multi-demo metric aggregation | `warp-metrics --all-demos` | Macro per-demo summary; does not claim the paper repository's global NNAD/PCAV pass |
| Ground-truth policy replay | `warp-rollout-policy --use-gt-action` | Verified on canonical and supplied sample HDF5s |
| Websocket policy rollout | `warp-rollout-policy` | Frozen msgpack protocol, chunked actions, metadata-selected proprio/camera payloads |
| Physical robot rollout | `external/rby1_teleop` | Robot/hardware responsibility stays in the submodule; later deployment code is out of WARP scope |

The local HDF5 writer adds `targets/` and `diagnostics/` alongside the original
robomimic fields. Those groups make portable metric validation possible without
copying analytical kinematics or requiring the raw CSV a second time. Existing
consumers can ignore them.

## Numerical-parity boundary

The canonical 420-frame input cache was checked value-for-value against the
frozen bone-to-action conversion. Against the archival Python converter, the
licensed Rust session currently matches both XHand trajectories exactly and
the 56D end-effector action to `1.2e-7` maximum absolute error. Its redundant
body-joint/base allocation is not bit-exact (maximum joint-action difference
`0.1201 rad` on this sequence), so this repository must not be used to silently
replace an already reported paper table. The archival converter remains the
authority for exact historical numbers; this repository is the maintained,
protected-source-free workflow and validation surface.

This repository does not contain the paper's separate aggregation, cluster
submission, and figure-production repository. In particular, global NNAD and
PCAV must not be computed independently per shard and averaged.
