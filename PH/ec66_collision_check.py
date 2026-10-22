"""EC66 manual-mode approximate collision guard; no Swift/PyBullet collision flags.

Uses the Week 5/6 line-vs-box-face idea (segment/box slab intersection).
Robot links are approximated by thickened centreline segments.  This is NOT
mesh-level collision detection and must not be treated as a safety-rated guard.
"""
import numpy as np
from spatialmath import SE3

# Conservative visual proxies; tune against the actual STL and robot geometry.
LINK_RADIUS_M = 0.045
EXTRA_MARGIN_M = 0.012
TOOL_EXTENSION_M = 0.075
# Skip the mounting/base segment: it is designed to touch the linear platform.
SKIP_BASE_SEGMENTS = 1


def _segment_hits_box(p0, p1, centre, rotation, half_size):
    """Finite line-segment / oriented-box intersection via the slab method."""
    a = (np.asarray(p0) - centre) @ rotation
    b = (np.asarray(p1) - centre) @ rotation
    d = b - a
    t_enter, t_exit = 0.0, 1.0
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if abs(a[k]) > half_size[k]:
                return False
        else:
            t0 = (-half_size[k] - a[k]) / d[k]
            t1 = (half_size[k] - a[k]) / d[k]
            t_enter = max(t_enter, min(t0, t1))
            t_exit = min(t_exit, max(t0, t1))
            if t_enter > t_exit:
                return False
    return True


def _box(name, pose, dimensions, margin=0.0):
    t = np.asarray(SE3(pose).A, dtype=float)
    return (name, t[:3, 3], t[:3, :3], np.asarray(dimensions, dtype=float) / 2 + margin)


class EC66CollisionChecker:
    def __init__(self, scenery, general_bucket_pose=None):
        self.scenery = scenery
        self.general_bucket_pose = general_bucket_pose
        self.last_hit = None

    def obstacles(self):
        scene = self.scenery
        # Printer proxy in printer-local frame. Calibrate these bounds against STL.
        # Front opening is not modelled; manual jog near the interior may be blocked.
        boxes = []
        for i, printer in enumerate(scene.printers, 1):
            boxes.append(_box(f'Printer {i} housing (approx.)',
                              printer.pose @ SE3(0.20, 0.20, 0.20),
                              (0.40, 0.40, 0.40)))
        # Approximate tabletop; intentionally exclude the robot's rail/mount.
        boxes.append(_box('Printer table top', scene.table.pose @ SE3(0.4, 0.0, 0.725),
                          (1.7, 0.85, 0.05)))
        if self.general_bucket_pose is not None:
            # Floor and four walls, not a solid cuboid: the bucket is open at the top.
            T = SE3(self.general_bucket_pose)
            width, wall, height = 0.24, 0.02, 0.15
            off = width / 2 - wall / 2
            boxes.append(_box('General bucket floor', T @ SE3(0, 0, -0.045),
                              (width, width, 0.02)))
            for axis in (-1, 1):
                boxes.append(_box('General bucket wall', T @ SE3(0, axis * off, 0.04),
                                  (width, wall, height)))
                boxes.append(_box('General bucket wall', T @ SE3(axis * off, 0, 0.04),
                                  (wall, width, height)))
        return boxes

    def check(self, q, rail_position):
        """Check a candidate pose without permanently moving the robot or rail."""
        scene = self.scenery
        robot = scene.robot
        old_base = SE3(robot.base)
        try:
            station = scene.platform.get_station_pose(float(rail_position))
            robot.base = station @ scene.robot_offset
            frames = robot.fkine_all(np.asarray(q, dtype=float))
            pts = [np.asarray(frame.t, dtype=float) for frame in frames]
            if len(pts) >= 2:
                z = np.asarray(frames[-1].R, dtype=float)[:, 2]
                pts.append(pts[-1] + TOOL_EXTENSION_M * z)
            for idx, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
                if idx < SKIP_BASE_SEGMENTS:
                    continue
                for name, c, R, half in self.obstacles():
                    if _segment_hits_box(a, b, c, R, half + LINK_RADIUS_M + EXTRA_MARGIN_M):
                        self.last_hit = f'{name} (link {idx})'
                        return self.last_hit
            self.last_hit = None
            return None
        finally:
            robot.base = old_base

    def swept_check(self, q0, rail0, q1, rail1, samples=5):
        for t in np.linspace(0.0, 1.0, max(2, samples)):
            hit = self.check((1-t)*q0 + t*q1, (1-t)*rail0 + t*rail1)
            if hit:
                return hit
        return None
