"""TX2-60 collision tests, independent of Swift/PyBullet.

Existing public API is retained.  The additional capsule-vs-oriented-box
check is continuous in space (no obstacle point-grid gaps).  This is a
geometric approximation, not certified robot safety or mesh collision.
"""
import numpy as np
from spatialmath import SE3


def _link_frames(robot, q, tool_length=0.12):
    frames = robot.fkine_all(q)
    points = [np.asarray(T.t, dtype=float) for T in frames]
    if not points:
        return []
    R = np.asarray(frames[-1].A[:3, :3], dtype=float)
    points.append(points[-1] + R @ np.array([0.0, 0.0, tool_length]))
    return list(zip(points[:-1], points[1:]))


def link_ellipsoids(robot, q, radius=0.07, margin=0.02, tool_length=0.12):
    """Legacy link proxies, retained for compatibility with other PH callers."""
    ellipsoids = []
    for p0, p1 in _link_frames(robot, q, tool_length):
        axis = p1 - p0
        length = float(np.linalg.norm(axis))
        if length < 1e-3:
            continue
        x = axis / length
        helper = np.array([0., 0., 1.]) if abs(x[2]) < 0.9 else np.array([1., 0., 0.])
        y = np.cross(helper, x)
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        ellipsoids.append(((p0 + p1) / 2, np.column_stack([x, y, z]),
                           np.array([length / 2 + margin, radius + margin, radius + margin])))
    return ellipsoids


def cuboid_points(cuboid, spacing=0.03):
    """Legacy point sampling helper (not used by the upgraded checker)."""
    scale = np.asarray(cuboid.scale, dtype=float)
    axes = [np.linspace(-s / 2, s / 2, max(2, int(np.ceil(s / spacing)) + 1)) for s in scale]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    T = np.asarray(SE3(cuboid.T).A)
    return grid @ T[:3, :3].T + T[:3, 3]


def in_ellipsoid(points, centre, rot, semi):
    local = (np.asarray(points) - centre) @ rot
    return np.sum((local / semi) ** 2, axis=1) < 1.0


def _segment_aabb_distance_sq(a, b, half):
    """Exact minimum squared distance from a segment to an axis-aligned box.

    Squared distance is piecewise quadratic in segment parameter t.  Split at
    every box-face crossing and evaluate each interval's stationary point.
    """
    a, b, half = map(lambda x: np.asarray(x, dtype=float), (a, b, half))
    d = b - a
    cuts = [0.0, 1.0]
    for i in range(3):
        if abs(d[i]) < 1e-12:
            continue
        for boundary in (-half[i], half[i]):
            t = (boundary - a[i]) / d[i]
            if 0.0 < t < 1.0:
                cuts.append(float(t))
    cuts = sorted(set(cuts))

    def dist2(t):
        outside = np.maximum(np.abs(a + t * d) - half, 0.0)
        return float(outside @ outside)

    best = min(dist2(t) for t in cuts)
    if best <= 0:
        return 0.0
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        mid = (lo + hi) / 2
        p = a + mid * d
        below = p < -half
        above = p > half
        active = below | above
        if not np.any(active):
            return 0.0
        bound = np.where(below, -half, half)
        da = d[active]
        denom = float(da @ da)
        if denom < 1e-18:
            continue
        t = -float(da @ (a[active] - bound[active])) / denom
        best = min(best, dist2(float(np.clip(t, lo, hi))))
    return best


def capsule_hits_cuboid(p0, p1, cuboid, radius):
    """Capsule centreline vs oriented cuboid; True if separation <= radius."""
    T = np.asarray(SE3(cuboid.T).A, dtype=float)
    R, origin = T[:3, :3], T[:3, 3]
    a = R.T @ (np.asarray(p0, dtype=float) - origin)
    b = R.T @ (np.asarray(p1, dtype=float) - origin)
    half = np.asarray(cuboid.scale, dtype=float).reshape(3) / 2
    return _segment_aabb_distance_sq(a, b, half) <= float(radius) ** 2


def in_collision(robot, q, obstacles, radius=0.07, margin=0.02,
                 tool_length=0.12, tool_radius=0.035, **kwargs):
    """Check TX2-60 arm capsules and gripper against registered Cuboids.

    The last segment is the tool extension, with a smaller proxy radius so
    the gripper may enter an open bin without falsely hitting its walls.
    No collision can be detected for obstacles missing from `obstacles`.
    """
    if not obstacles:
        return False
    segments = _link_frames(robot, q, tool_length=tool_length)
    for index, (p0, p1) in enumerate(segments):
        if np.linalg.norm(p1 - p0) < 1e-8:
            continue
        r = (tool_radius if index == len(segments) - 1 else radius) + margin
        for obstacle_index, obstacle in enumerate(obstacles):
            # Allowed mounting contact: TX2-60 base with supporting table
            if index == 0 and obstacle_index == 0:
                continue
            if capsule_hits_cuboid(p0, p1, obstacle, r):
                # Log only when the first collision pair changes, not every Swift frame.
                pair = (index, obstacle_index)
                if getattr(in_collision, "_last_pair", None) != pair:
                    centre = np.asarray(SE3(obstacle.T).t, dtype=float)
                    scale = np.asarray(obstacle.scale, dtype=float)
                    print(f"[TX2 HIT DETAIL] segment={index}/{len(segments)-1} "
                          f"obstacle={obstacle_index}/{len(obstacles)-1} "
                          f"center={np.round(centre, 3)} "
                          f"scale={np.round(scale, 3)} "
                          f"p0={np.round(p0, 3)} p1={np.round(p1, 3)} "
                          f"capsule_radius={r:.3f}", flush=True)
                    in_collision._last_pair = pair
                return True
    in_collision._last_pair = None
    return False
