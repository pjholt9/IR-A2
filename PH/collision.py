import numpy as np
from spatialmath import SE3


# I approximate each arm segment with a padded ellipsoid for fast collision checks.
def link_ellipsoids(robot, q, radius=0.07, margin=0.02, tool_length=0.12):
    """Return (centre, rotation, semi_axes) ellipsoids enclosing each arm link for configuration q."""
    frames = robot.fkine_all(q)
    points = [np.asarray(T.t, dtype=float) for T in frames]
    # Extend past the last frame to cover the gripper
        # Extend the last frame along the tool axis so the gripper is covered too.
    points.append(frames[-1].A[:3, :3] @ np.array([0, 0, tool_length]) + points[-1])

    ellipsoids = []
    for p0, p1 in zip(points[:-1], points[1:]):
        axis = p1 - p0
                # The segment direction becomes the long ellipsoid axis; skip zero-length spans.
        length = np.linalg.norm(axis)
        if length < 1e-3:
            continue
        x = axis / length
                # Pick a helper away from x so the cross products form a stable local frame.
        helper = np.array([0, 0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0, 0])
        y = np.cross(helper, x)
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        rot = np.column_stack([x, y, z])
        semi = np.array([length / 2 + margin, radius + margin, radius + margin])
        ellipsoids.append(((p0 + p1) / 2, rot, semi))
    return ellipsoids


# Sample each obstacle in its own frame, then move those points into world coordinates.
def cuboid_points(cuboid, spacing=0.03):
    """Sample points throughout a spatialgeometry Cuboid in world coordinates."""
    scale = np.asarray(cuboid.scale, dtype=float)
    axes = [np.linspace(-s / 2, s / 2, max(2, int(np.ceil(s / spacing)) + 1)) for s in scale]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        # The cuboid transform rotates and translates its local grid into the workcell.
    T = np.asarray(SE3(cuboid.T).A)
    return grid @ T[:3, :3].T + T[:3, 3]


# Test points in the ellipsoid's local frame, where the standard equation applies.
def in_ellipsoid(points, centre, rot, semi):
    """Boolean mask of points inside the ellipsoid (algebraic test (x/a)^2+(y/b)^2+(z/c)^2 < 1)."""
        # Subtract the centre and rotate into local coordinates before scaling by each axis.
    local = (points - centre) @ rot
    return np.sum((local / semi) ** 2, axis=1) < 1.0


# Check all sampled obstacles against the arm's link proxies and stop on the first hit.
def in_collision(robot, q, obstacles, **kwargs):
    """True if any link ellipsoid at configuration q intersects any cuboid obstacle."""
    ellipsoids = link_ellipsoids(robot, q, **kwargs)
    for obstacle in obstacles:
        pts = cuboid_points(obstacle)
        if any(in_ellipsoid(pts, c, r, s).any() for c, r, s in ellipsoids):
            return True
    return False
