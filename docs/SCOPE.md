# Frozen scope

The inclusion boundary is the submitted WARP implementation as of 2026-05-30.
The canonical source checkpoint is `SEW-Geometric-Teleop@4f0fa5e`. The later
curated view added by `80f92b6` is useful as a file map, but it does not expand
the project boundary.

Included:

- Frozen WARP configuration and variant definitions.
- Offline human-motion to RBY1/XHand retargeting orchestration.
- Paper baselines, metrics integration, rendering, and reproducibility tools
  that existed at the cutoff.
- One explicitly allowlisted small reproduction fixture, if needed.

Excluded:

- Analytic or patented solver implementation source.
- Work first developed after 2026-05-30.
- Robot implementation and hardware code, which belong in `rby1_teleop`.
- Generated datasets and result artifacts.

Bug fixes after the cutoff may be applied only as isolated maintenance commits
that document whether they preserve or intentionally change paper parity.
