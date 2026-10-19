"""STL-backed printed parts positioned relative to printer.plate_reference.

Set the STL filenames in PART_STL_FILES below. Assumptions:
- STL units are millimetres, converted to metres via scale=[0.001]*3.
- STL Z is the vertical axis and the model is in its intended upright orientation.
- The part origin is automatically centered in XY and placed at its bottom in Z.
"""
from pathlib import Path
import numpy as np
from spatialmath import SE3
from spatialgeometry import Mesh

# EDIT THESE TWO FILE NAMES ONLY
PART_STL_FILES = {
    "barrel": "models/Barrel - A2.STL",
    "body": "models/Sled Body Front - A2.STL",
}

# Physical heights from your slicer, metres.
PART_HEIGHTS = {"barrel": 0.240, "body": 0.17498}


# Bambu Studio dimensions, metres. The STL-local origin is the footprint corner.
# Pick XYZ is derived from the selected part, never from GUI screenshot values.
PART_CONFIG = {
    "barrel": {"width": 0.050, "depth": 0.04392, "height": 0.240,
               "pick_y": 0.0, "marker_color": [1.0, 0.0, 0.0, 1.0]},
    "body": {"width": 0.07560, "depth": 0.07906, "height": 0.17498,
             "pick_y": 0.0, "marker_color": [0.0, 0.3, 1.0, 1.0]},
}


def centered_part_offset(kind, center_xy=(0.128, 0.128)):
    """Offset relative to printer.bambu_origin (NOT plate_reference)."""
    c = PART_CONFIG[kind]
    return SE3(center_xy[0] - c["width"] / 2,
               center_xy[1] - c["depth"] / 2, 0.0)


def pick_local_pose(kind):
    """STL-local marker pose: front face Y=0, X midpoint, Z=2/3 height."""
    c = PART_CONFIG[kind]
    return SE3(c["width"] / 2, c["pick_y"], 2.0 * c["height"] / 3.0)

# Optional vertical clearance if plate_reference is not on the plate surface.
DEFAULT_PLATE_Z = 0.0


class PrintedPart:
    def __init__(self, plate_reference, kind="barrel", offset=None,
                 plate_z=DEFAULT_PLATE_Z, stl_path=None):
        if kind not in PART_STL_FILES:
            raise ValueError(f"Unknown part kind: {kind}")
        self.kind = kind
        self.plate_reference = SE3(plate_reference.A)
        self.offset = SE3() if offset is None else SE3(offset.A)
        self.plate_z = float(plate_z)
        self.height = PART_HEIGHTS[kind]
        self._attached_to = None
        self._T_ee_part = None
        self.isAttached = False
        self.isAttached = False

        filename = stl_path or PART_STL_FILES[kind]
        path = Path(filename)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if not path.is_file():
            raise FileNotFoundError(f"STL not found: {path}")

        # Mesh centre/scale behaviour can depend on the installed
        # spatialgeometry version; verify placement visually in Swift.
        self.mesh = Mesh(filename=str(path), scale=[0.001, 0.001, 0.001],
                         pose=SE3(), color=([0.12, 0.12, 0.13, 1]
                                           if kind == "barrel"
                                           else [0.36, 0.23, 0.17, 1]))
        self.mesh._collision = False
        self._env = None
        self.update_pose()

    @property
    def pose(self):
        """World pose of the STL's local origin."""
        return self.plate_reference @ SE3(0, 0, self.plate_z) @ self.offset

    @property
    def pick_pose(self):
        return self.pose @ pick_local_pose(self.kind)

    @property
    def grasp_pose(self):
        """Nominal 2/3-height grasp reference (assumes STL origin at bottom)."""
        return self.pose @ SE3(0, 0, self.height * 2.0 / 3.0)

    def update_pose(self, offset=None, plate_reference=None):
        if offset is not None:
            self.offset = SE3(offset.A)
        if plate_reference is not None:
            self.plate_reference = SE3(plate_reference.A)
        self.mesh.T = np.asarray(self.pose.A, dtype=float)

    def add_to_env(self, env):
        if self._env is not None:
            raise RuntimeError("Part already added to an environment")
        env.add(self.mesh)
        self._env = env

    def attach_to(self, robot, grasp_world_pose=None, tolerance_m=0.001):
        """Attach only if EE-to-grasp POSITION error is strictly < tolerance.

        Returns bool; a failed check does not modify the mesh or attachment.
        This validates simulated positioning, not physical gripping/contact.
        """
        if self.isAttached or self._attached_to is not None:
            print("ATTACH FAILED: part is already attached")
            return False
        if tolerance_m <= 0:
            raise ValueError("tolerance_m must be positive")
        T_ee = robot.fkine(robot.q)
        T_grasp = self.pick_pose if grasp_world_pose is None else grasp_world_pose
        error_m = float(np.linalg.norm(T_ee.t - T_grasp.t))
        print(f"Grasp position error: {error_m * 1000:.4f} mm")
        if not np.isfinite(error_m) or error_m >= tolerance_m:
            print(f"ATTACH FAILED: grasp error exceeds tolerance "
                  f"({error_m * 1000:.4f} mm; required < {tolerance_m * 1000:.3f} mm)")
            return False
        T_part = SE3(np.asarray(self.mesh.T, dtype=float))
        self._T_ee_part = T_ee.inv() @ T_part
        self._attached_to = robot
        self.isAttached = True
        self.follow_ee()
        print("ATTACH SUCCESS: part locked to EE")
        return True

    def follow_ee(self):
        """Call once after EVERY robot.q update while carrying the part."""
        if self._attached_to is None:
            return
        T_world_part = self._attached_to.fkine(self._attached_to.q) @ self._T_ee_part
        self.mesh.T = np.asarray(T_world_part.A, dtype=float)

    def detach(self):
        """Freeze mesh at its current world pose; no automatic placement."""
        self.follow_ee()
        self._attached_to = None
        self._T_ee_part = None
        self.isAttached = False
