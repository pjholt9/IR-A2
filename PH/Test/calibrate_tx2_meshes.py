"""One-off calibration of the TX2-60 STL frames against its DH model."""

from itertools import product
from math import pi
from pathlib import Path

import numpy as np
from spatialmath import SE3

if __package__:
    from ..tx2_60 import TX2_60
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tx2_60 import TX2_60


MESH_DIR = Path(__file__).resolve().parents[1] / "TX2_60meshes"
URDF_ORIGINS = (
    ((0, -0.375, 0), (0, 0, 0)),
    ((0, 0.375, 0), (0, 0, 0)),
    ((0, -0.665, 0.02), (0, 0, 0)),
    ((0, 0, 0), (pi / 2, 0, 0)),
    ((0, 0, 0.31), (-pi / 2, 0, 0)),
    ((0, -0.07, 0), (pi / 2, 0, 0)),
)
URDF_RPY_ORDER = "zyx"
WRIST_HEIGHT = 0.975
ANGLE_CHOICES = (0, pi / 2, -pi / 2)


def urdf_frames():
    """Return the seven cumulative URDF link frames at the zero pose."""
    frames = [SE3()]
    for xyz, rpy in URDF_ORIGINS:
        frames.append(
            frames[-1]
            * SE3(xyz)
            * SE3.RPY(rpy, order=URDF_RPY_ORDER)
        )
    return frames


def find_reference(robot):
    """Find the requested vertical configuration using the actual DH model."""
    candidates = (
        np.array([0, q2, q3, 0, 0, 0], dtype=float)
        for q2, q3 in product(ANGLE_CHOICES, repeat=2)
    )
    for q_ref in candidates:
        frames = robot.fkine_all(q_ref, old=False)
        wrist = frames[5].t
        tool_z = frames[6].R[:, 2]
        if (
            np.isclose(wrist[2], WRIST_HEIGHT, atol=1e-9)
            and np.allclose(tool_z, [0, 0, 1], atol=1e-9)
        ):
            return q_ref, frames
    raise RuntimeError("No requested vertical q_ref was found in the DH search.")


def rotation_error(first, second):
    relative = first.T @ second
    cosine = np.clip((np.trace(relative) - 1) / 2, -1, 1)
    return float(np.arccos(cosine))


def choose_t_fix(dh_frames, frames_urdf):
    """Choose the quarter-turn candidate aligned with DH link 0 at q1=0."""
    candidates = [
        SE3.Rz(k * pi / 2) * SE3.Rx(-pi / 2)
        for k in range(4)
    ]
    return min(
        candidates,
        key=lambda candidate: rotation_error(
            dh_frames[1].R,
            (candidate * frames_urdf[1]).R,
        ),
    )


def print_transform(name, transform):
    xyz = np.round(transform.t, 6)
    rpy = np.round(transform.rpy(order=URDF_RPY_ORDER), 6)
    print(f"{name}: xyz={xyz.tolist()}, rpy_zyx={rpy.tolist()}")


def print_offset_constant(index, transform):
    xyz = np.round(transform.t, 6)
    rpy = np.round(transform.rpy(order=URDF_RPY_ORDER), 6)
    print(
        f"    SE3({xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f})"
        f" * SE3.RPY([{rpy[0]:.6f}, {rpy[1]:.6f}, {rpy[2]:.6f}], "
        f'order="{URDF_RPY_ORDER}"),  # {index}'
    )


def main():
    missing = [
        filename
        for filename in TX2_60.MESH_FILES
        if not (MESH_DIR / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing TX2-60 mesh file(s): " + ", ".join(missing)
        )

    tx2_60 = TX2_60()
    robot = tx2_60.create_robot()
    q_ref, dh_frames = find_reference(robot)
    frames_urdf = urdf_frames()
    t_fix = choose_t_fix(dh_frames, frames_urdf)
    offsets = [
        dh_frames[index].inv() * t_fix * frames_urdf[index]
        for index in range(7)
    ]

    print(f"q_ref: {np.round(q_ref, 6).tolist()}")
    print_transform("T_fix", t_fix)
    print(f"T_fix matrix:\n{np.array2string(t_fix.A, precision=6)}")
    print(f"URDF RPY order: {URDF_RPY_ORDER}")
    print("MESH_OFFSETS = (")
    for index, offset in enumerate(offsets):
        print_offset_constant(index, offset)
    print(")")

    print("Alignment differences to retain/report:")
    print(
        "  DH a1=0.05 m has no explicit URDF counterpart; "
        "offset[1] includes its -0.05 m x correction."
    )
    print(
        "  URDF frame 2 is at the base origin, while DH frame 2 is at "
        f"{np.round(dh_frames[2].t, 6).tolist()} m."
    )
    print(
        "  DH d6=0.080 m vs URDF d6=0.070 m; "
        f"mapped frame 6 ends at {np.round((t_fix * frames_urdf[6]).t, 6).tolist()} m "
        f"rather than {np.round(dh_frames[6].t, 6).tolist()} m."
    )


if __name__ == "__main__":
    main()
