"""Retarget one or more frame streams into one frozen RBY1 HDF5 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .offline_validate import run_validation
from .variants import DEFAULT_REGISTRY, variant_names


def _label(path: Path) -> str:
    name = path.name
    for suffix in (".frames.npz", ".npz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def build_dataset(
    frames: list[str | Path],
    *,
    variant: str,
    output_dir: str | Path,
    max_frames: int | None = None,
    registry: str | Path = DEFAULT_REGISTRY,
    overwrite: bool = False,
) -> dict:
    """Build a multi-demo paper-format dataset from solver-neutral streams."""
    paths = [Path(path) for path in frames]
    if not paths:
        raise ValueError("at least one frame stream is required")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"frame stream(s) not found: {', '.join(missing)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = output_dir / "robot_data.hdf5"
    if hdf5_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{hdf5_path} already exists; pass --overwrite to replace it"
            )
        hdf5_path.unlink()

    summaries = []
    for index, path in enumerate(paths):
        demo_key = f"demo_{index}"
        episode_dir = output_dir / "episodes" / f"{demo_key}_{_label(path)}"
        summary = run_validation(
            variant,
            frames=path,
            output_dir=episode_dir,
            max_frames=max_frames,
            registry=registry,
            demo_key=demo_key,
            hdf5_path=hdf5_path,
        )
        summaries.append(summary)

    result = {
        "variant": variant,
        "hdf5": str(hdf5_path),
        "demos": len(summaries),
        "frames": int(sum(item["frames"] for item in summaries)),
        "episodes": summaries,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+")
    parser.add_argument("--variant", required=True, choices=variant_names())
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = build_dataset(
        args.frames,
        variant=args.variant,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        registry=args.registry,
        overwrite=args.overwrite,
    )
    print(json.dumps({key: result[key] for key in ("variant", "hdf5", "demos", "frames")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
