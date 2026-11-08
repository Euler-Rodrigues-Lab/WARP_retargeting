# Precomputed licensed-mode replays

These files contain the derived output of the five frozen SEW variants on the
canonical `washing_dishes_R_004__A299` frame stream. They allow robot and human
overlay replay without installing the licensed `geo_kin` backend.

| File suffix | Frozen variant |
|---|---|
| `warp` | WARP (`ours` in the frozen experiment scripts) |
| `warp_no_joint_limits` | Archival SEED example preset: joint limits and spring-damper base off |
| `c_sew` | C-SEW/TCP baseline without functional offset |
| `warp_no_spring` | WARP without spring-damper base |
| `sew_mimic` | Pure SEW-mimic baseline |

For example:

```bash
uv sync --extra runtime
uv run warp-replay-hdf5 \
  fixtures/retargeted/washing_dishes_warp.hdf5 --loop
```

The human capsule overlay is enabled by default. Use `--no-human-overlay` to
hide it, `--mode action` for the generated commands, and `H` to toggle the
overlay while the viewer is open.

These are derived trajectories, not implementations of the licensed solver.
Retargeting a new recording with a SEW variant still requires a licensed
`geo_kin` wheel. The recorded solve timings are informative generation
metadata and should not be treated as a cross-machine performance benchmark.

See `manifest.json` for source and artifact hashes.
