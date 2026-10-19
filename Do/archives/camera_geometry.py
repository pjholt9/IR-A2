"""Camera position utilities. All translations in metres.

IMPORTANT: CAMERA_WORLD_OFFSET is expressed in WORLD axes, NOT EE-local axes.
Camera rotation is not calibrated here; don't use EE rotation as camera rotation.
"""
import numpy as np

CAMERA_WORLD_OFFSET = np.array([-0.350, 0.0, 0.0], dtype=float)


def camera_world_position(robot):
    """Return camera optical-centre position assuming calibrated offset is exact."""
    T_world_ee = robot.fkine(robot.q)
    return np.asarray(T_world_ee.t, dtype=float).reshape(3) + CAMERA_WORLD_OFFSET


def capture_geometry(robot, plate_reference=None):
    """Snapshot the capture-time pose BEFORE robot retreats.

    Camera orientation / T_world_camera is deliberately omitted until calibrated.
    """
    T_world_ee = robot.fkine(robot.q)
    data = {
        "world_T_ee_at_capture": np.asarray(T_world_ee.A).tolist(),
        "camera_world_xyz_m": camera_world_position(robot).tolist(),
        "camera_offset_world_xyz_m": CAMERA_WORLD_OFFSET.tolist(),
        "camera_rotation_calibrated": False,
        "camera_world_transform": None,
    }
    if plate_reference is not None:
        data["world_T_plate_reference"] = np.asarray(plate_reference.A).tolist()
    return data
