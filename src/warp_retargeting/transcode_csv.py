"""Transcode a frozen SEED bone CSV/CSV.GZ into a solver-neutral frame stream."""

from __future__ import annotations

import argparse

from geo_kin_core.frames import save_frames
from rby1_teleop.input import OfflineCSVAdapter


def transcode(csv_file, output, *, fps: float = 30.0, monolith_path=None):
    """Sample from t=0 through the final 30 Hz tick, matching the paper converter."""
    source = OfflineCSVAdapter(csv_file, loop=False, monolith_path=monolith_path)
    count = int(source.duration * fps) + 1
    frames = []
    for index in range(count):
        frame = source.frame_at_time(index / fps)
        if frame is None:
            break
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"no valid bone frames read from {csv_file}")
    return save_frames(
        output,
        frames,
        fps=fps,
        source=str(csv_file),
        notes="Frozen SEED bone CSV sampled with the WARP paper timing contract",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--monolith-path", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    path = transcode(
        args.csv_file, args.output, fps=args.fps,
        monolith_path=args.monolith_path,
    )
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
