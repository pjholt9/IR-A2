"""Independent vision processing prototype: load depth and RANSAC fit a plane.

This is NOT yet a grasp-pose estimator. Depth units, camera intrinsics, RGB/depth
registration and camera extrinsics must be verified before estimating world poses.
Supports .npy and single-channel 16-bit .png depth, and .exr if OpenCV EXR is enabled.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def load_depth(path: str | Path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
    else:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install OpenCV: pip install opencv-python") from exc
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Could not read depth: {path}. For EXR, check OpenCV EXR support.")
    if arr.ndim != 2:
        raise ValueError(f"Expected single-channel depth, got {arr.shape}")
    return arr


def depth_to_xyz(depth, fx, fy, cx, cy, units_to_m):
    """Back-project registered depth pixels into camera coordinates (metres).

    Assumes depth values represent optical-axis Z, not radial range.
    """
    if fx <= 0 or fy <= 0 or units_to_m <= 0:
        raise ValueError("Focal lengths and depth unit scale must be positive")
    depth = np.asarray(depth, dtype=np.float64)
    v, u = np.indices(depth.shape)
    z = depth * units_to_m
    good = np.isfinite(z) & (z > 0)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.column_stack((x[good], y[good], z[good])), good


def ransac_plane(points, threshold_m=0.006, iterations=800, seed=11):
    """Fit ax+by+cz+d=0, ||[a,b,c]||=1, to 3D points."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        raise ValueError("Expected N>=3 3D points")
    if threshold_m <= 0:
        raise ValueError("threshold_m must be positive")
    rng = np.random.default_rng(seed)
    best = np.zeros(len(pts), dtype=bool)
    for _ in range(iterations):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        d = -normal @ a
        mask = np.abs(pts @ normal + d) <= threshold_m
        if mask.sum() > best.sum():
            best = mask
    if best.sum() < 3:
        raise ValueError("RANSAC could not find a plane")
    # Refine plane on consensus via SVD.
    centroid = pts[best].mean(axis=0)
    _, _, vh = np.linalg.svd(pts[best] - centroid, full_matrices=False)
    normal = vh[-1]
    d = -normal @ centroid
    refined = np.abs(pts @ normal + d) <= threshold_m
    return {"normal_camera": normal.tolist(), "d_m": float(d),
            "inlier_count": int(refined.sum()), "total_count": len(pts),
            "inlier_fraction": float(refined.mean()), "inlier_mask": refined}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", required=True, type=Path)
    parser.add_argument("--fx", required=True, type=float)
    parser.add_argument("--fy", required=True, type=float)
    parser.add_argument("--cx", required=True, type=float)
    parser.add_argument("--cy", required=True, type=float)
    parser.add_argument("--depth-scale", required=True, type=float,
                        help="Metres per raw depth unit; e.g. 0.001 for millimetres")
    parser.add_argument("--threshold-mm", type=float, default=6.0)
    parser.add_argument("--max-points", type=int, default=60000)
    args = parser.parse_args()
    depth = load_depth(args.depth)
    pts, _ = depth_to_xyz(depth, args.fx, args.fy, args.cx, args.cy, args.depth_scale)
    if len(pts) > args.max_points:
        pts = pts[np.random.default_rng(11).choice(len(pts), args.max_points, replace=False)]
    result = ransac_plane(pts, threshold_m=args.threshold_mm / 1000)
    result.pop("inlier_mask")
    print(json.dumps(result, indent=2))
    print("NOTE: fitted dominant plane is not necessarily the build plate. Verify visually.")


if __name__ == "__main__":
    main()
