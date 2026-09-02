#!/usr/bin/env python3
"""Fail when protected, post-WARP, generated, or foreign-author material enters Git."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTECTED_PATHS = (
    re.compile(r"geometric_kinematics.*\.py$"),
    re.compile(r"geometric_subproblems.*\.py$"),
    re.compile(r"sew_stereo.*\.py$"),
    re.compile(r"constrained_sew\.py$"),
    re.compile(r"rby1(_with_xhand)?_sew_solver\.py$"),
)
POST_WARP_PATH_TERMS = {
    "aria_egoposer", "egoposer", "perception_pipeline", "partial_mocap",
    "firm_grasp", "seed_recon", "adapt3r", "fastfs", "egoengine",
}
GENERATED_SUFFIXES = {
    ".hdf5", ".h5", ".csv", ".gz", ".mp4", ".mov", ".png", ".jpg", ".npz",
}
ALLOWLISTED_DATA = {
    "fixtures/seed/washing_dishes_R_004__A299.csv.gz",
    "fixtures/seed/washing_dishes_R_004__A299.frames.npz",
}
FORBIDDEN_CODE = (
    "from projects.", "import projects.", "GEO_TELEOP_MONOLITH",
    "/home/", "/coc/", "/media/",
    "FR_IK_SP_3_Arm_Deviation", "FR_TCP_Centroid_Alignment",
)
CODE_ROOTS = ("src/", "experiments/", "scripts/")
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".sh"}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def has_commits() -> bool:
    return subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def validate() -> list[str]:
    errors: list[str] = []
    tracked = [line for line in git("ls-files").splitlines() if line]
    for rel in tracked:
        lowered = rel.lower()
        if any(pattern.search(rel) for pattern in PROTECTED_PATHS):
            errors.append(f"protected solver path: {rel}")
        if any(term in lowered for term in POST_WARP_PATH_TERMS):
            errors.append(f"post-WARP path: {rel}")
        if Path(rel).suffix.lower() in GENERATED_SUFFIXES and rel not in ALLOWLISTED_DATA:
            errors.append(f"generated/data file is not allowlisted: {rel}")
        path = ROOT / rel
        if (
            rel != "scripts/check_repository_scope.py"
            and rel.startswith(CODE_ROOTS)
            and path.suffix.lower() in TEXT_SUFFIXES
            and path.is_file()
        ):
            text = path.read_text(errors="replace")
            for marker in FORBIDDEN_CODE:
                if marker in text:
                    errors.append(f"forbidden code marker {marker!r}: {rel}")

    if has_commits():
        authors = {line for line in git("log", "--format=%an").splitlines() if line}
        bodies = git("log", "--format=%B")
    else:
        authors = set()  # Empty repository before its first commit.
        bodies = ""
    foreign = authors - {"kczttm"}
    if foreign:
        errors.append(f"foreign Git author(s): {', '.join(sorted(foreign))}")
    if re.search(r"(?im)^co-authored-by:", bodies):
        errors.append("co-author trailer found; repository author must be only kczttm")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("Repository scope check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Repository scope check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
