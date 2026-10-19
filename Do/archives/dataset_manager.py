"""Import an iPhone RGB-D/spatial capture without modifying the original data.

Example (from IR-A2 root):
 python Do/dataset_manager.py --sample center --source "C:\\Users\\...\\Downloads\\capture01"

Accepts directories or one or more files. Keeps original filenames and raw bytes.
No claim is made that depth and RGB are aligned until inspected/calibrated.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

DATA_ROOT = Path(__file__).resolve().parent / "datasets" / "barrel"
SAMPLES = ("center", "front_left", "front_right", "back_left", "back_right")
SAMPLE_CENTERS_M = {"center": [0.128, 0.128, 0.0]}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_capture(sample: str, sources: list[Path], root: Path = DATA_ROOT, overwrite=False):
    if sample not in SAMPLES:
        raise ValueError(f"Unknown sample {sample!r}; choose from {SAMPLES}")
    paths = [p.expanduser().resolve() for p in sources]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    source_files = []
    for p in paths:
        if p.is_dir():
            source_files.extend(f for f in p.rglob("*") if f.is_file())
        elif p.is_file():
            source_files.append(p)
    if not source_files:
        raise ValueError("No files found in source")
    target = root / sample
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(f"{target} already contains data; choose --overwrite to replace")
    # Stage files before changing destination, preserving raw source folder hierarchy.
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    used = set()
    for p in source_files:
        parent = next((s for s in paths if s.is_dir() and p.is_relative_to(s)), None)
        relative = p.relative_to(parent) if parent else Path(p.name)
        relative = Path("raw") / relative
        if relative.as_posix().lower() in used:
            raise ValueError(f"Duplicate destination path: {relative}")
        used.add(relative.as_posix().lower())
        dst = target / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not overwrite:
            raise FileExistsError(dst)
        shutil.copy2(p, dst)
        manifest.append({"path": relative.as_posix(), "bytes": dst.stat().st_size,
                         "sha256": sha256_file(dst), "original_filename": p.name})
    meta_path = target / "metadata.json"
    old = {}
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeError):
            raise ValueError(f"Existing metadata invalid: {meta_path}")
    old.update({
        "schema_version": 1,
        "sample": sample,
        "part": "barrel",
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "nominal_part_center_bambu_m": SAMPLE_CENTERS_M.get(sample),
        "files": manifest,
        "camera_world_offset_m": [-0.350, 0.0, 0.0],
        "camera_world_offset_frame": "world",
        "camera_intrinsics": old.get("camera_intrinsics"),
        "camera_to_ee_rotation": old.get("camera_to_ee_rotation"),
        "world_T_ee_at_capture": old.get("world_T_ee_at_capture"),
        "camera_world_xyz_m": old.get("camera_world_xyz_m"),
        "rgb_depth_alignment_verified": False,
        "depth_units": None,
        "notes": "Original exports are preserved in raw/. Fill calibration fields only from measured data.",
    })
    meta_path.write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding="utf-8")
    return target, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=SAMPLES, required=True)
    parser.add_argument("--source", type=Path, nargs="+", required=True,
                        help="Capture folder or individual exported files")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        folder, files = import_capture(args.sample, args.source, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Imported {len(files)} files to {folder}")
    for f in files:
        print(f"  {f['path']} ({f['bytes']} bytes)")
    print("Metadata:", folder / "metadata.json")
    print("Next: inspect RGB/depth formats and calibration before running RANSAC.")


if __name__ == "__main__":
    main()
