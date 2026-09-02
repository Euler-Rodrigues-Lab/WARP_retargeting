# Supplied SEED sample validation

Validated locally on 2026-09-02 with the licensed backend and the `c_sew`
variant. The source `.csv.gz` files stayed in `/home/ck/Downloads`; neither
they nor generated HDF5/NPZ output is tracked by this repository.

| Recording | Frames at 30 Hz |
|---|---:|
| `crouch_operating_cupboard_mid_out_R_003__A298_M` | 187 |
| `cutting_bread_R_003__A299_M` | 223 |
| `cutting_masterchiefstyle_R_003__A299_M` | 328 |
| `frying_pan_R_004__A299_M` | 267 |
| `grating_vegetables_R_003__A299_M` | 157 |
| `heavy_item_cupboard_high_out_R_003__A299_M` | 304 |
| `looting_cupboard_high_R_004__A297_M` | 326 |
| `making_friedeggs_R_005__A299_M` | 358 |

All eight were transcoded, retargeted in one 2,150-frame/multi-demo HDF5,
processed by the portable metric runner, replayed headlessly from recorded
actions, and exercised through ground-truth policy rollout. This is workflow
validation, not a paper-result claim: these mirrored recordings are not the
paper's documented evaluation subset, and C-SEW intentionally permits the
tracking/feasibility tradeoffs visible in the metrics.
