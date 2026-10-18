"""Check calibrated TX2-60 mesh transforms and optionally view them in Swift."""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from spatialgeometry import Axes
from swift import Swift

if __package__:
    from .calibrate_tx2_meshes import find_reference
    from .tx2_60 import TX2_60
else:
    from calibrate_tx2_meshes import find_reference
    from tx2_60 import TX2_60


REFERENCE_HEIGHT = 1.045
MAX_BASE_DISTANCE = 1.3
RANDOM_CONFIGURATION_COUNT = 10
RANDOM_SEED = 23
STEPS_PER_CONFIGURATION = 10


def test_configurations(tx2_60):
    q_ref, _ = find_reference(tx2_60.robot)
    configurations = [("q_ref", q_ref)]
    sweep_origin = np.zeros(tx2_60.robot.n)
    for joint in range(tx2_60.robot.n):
        for angle in (-np.pi / 4, np.pi / 4):
            q = sweep_origin.copy()
            q[joint] += angle
            sign = "minus" if angle < 0 else "plus"
            configurations.append((f"joint_{joint + 1}_{sign}_45deg", q))

    rng = np.random.default_rng(RANDOM_SEED)
    qlim = np.asarray(tx2_60.robot.qlim)
    for index in range(RANDOM_CONFIGURATION_COUNT):
        q = rng.uniform(qlim[0], qlim[1])
        configurations.append((f"random_{index + 1}", q))
    configurations.append(("q_standby", tx2_60.q_standby.copy()))
    return configurations


def load_mesh_vertices():
    mesh_dir = Path(__file__).resolve().parent / "TX2_60meshes"
    meshes = []
    for filename in TX2_60.MESH_FILES:
        path = mesh_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required TX2-60 mesh is missing: {path}")
        mesh = trimesh.load(path, force="mesh", process=False)
        meshes.append(np.asarray(mesh.vertices, dtype=float))
    return meshes


def world_vertices(tx2_60, q, local_vertices):
    frames = tx2_60.robot.fkine_all(q, old=False)
    transformed = []
    for index, vertices in enumerate(local_vertices):
        transform = (
            tx2_60.base * tx2_60.MESH_OFFSETS[0]
            if index == 0
            else frames[index] * tx2_60.MESH_OFFSETS[index]
        )
        transformed.append(
            (transform.R @ vertices.T).T + transform.t
        )
    return transformed


def verify_configuration(env, tx2_60, name, q, local_vertices):
    robot = tx2_60.robot
    robot.q = q
    env.step(0.02, render=False)

    frames = robot.fkine_all(q, old=False)
    for index, mesh in enumerate(tx2_60.link_meshes):
        expected = frames[index + 1] * tx2_60.MESH_OFFSETS[index + 1]
        if not np.allclose(mesh._wT, expected.A, atol=1e-8):
            raise AssertionError(
                f"{name}: Swift link mesh {index + 1} does not follow its DH frame"
            )

    expected_base = tx2_60.base * tx2_60.MESH_OFFSETS[0]
    if not np.allclose(tx2_60.base_mesh.T, expected_base.A, atol=1e-8):
        raise AssertionError(f"{name}: static base mesh moved with a joint")

    vertices = world_vertices(tx2_60, q, local_vertices)
    all_vertices = np.vstack(vertices)
    relative = all_vertices - tx2_60.base.t
    max_distance = float(np.abs(relative).max())
    if max_distance > MAX_BASE_DISTANCE:
        raise AssertionError(
            f"{name}: mesh bounds exceed +/-{MAX_BASE_DISTANCE} m "
            f"(observed {max_distance:.3f} m)"
        )

    if name == "q_ref":
        height = float(all_vertices[:, 2].max() - all_vertices[:, 2].min())
        if not np.isclose(height, REFERENCE_HEIGHT, atol=0.02):
            raise AssertionError(
                f"q_ref: expected about {REFERENCE_HEIGHT:.3f} m height, "
                f"observed {height:.3f} m"
            )

    print(
        f"{name}: mesh z=[{all_vertices[:, 2].min():.3f}, "
        f"{all_vertices[:, 2].max():.3f}] m, "
        f"max |base-relative xyz|={max_distance:.3f} m"
    )


def add_frame_axes(tx2_60):
    for link in tx2_60.robot.links:
        link.geometry = [*link.geometry, Axes(0.08)]


def animate_to(env, tx2_60, q):
    start = tx2_60.robot.q.copy()
    for q_step in np.linspace(start, q, STEPS_PER_CONFIGURATION):
        tx2_60.robot.q = q_step
        env.step(0.02)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the transform and bounds checks without opening Swift.",
    )
    args = parser.parse_args()

    local_vertices = load_mesh_vertices()
    tx2_60 = TX2_60()
    robot = tx2_60.create_robot()
    robot.control_mode = "p"
    add_frame_axes(tx2_60)

    env = Swift()
    env.launch(realtime=not args.headless, headless=args.headless)
    tx2_60.create_mesh_robot(env)
    if not tx2_60.using_meshes:
        raise RuntimeError("The TX2-60 mesh checks require all STL files.")
    env.add(Axes(0.1, pose=tx2_60.base))

    base_transform = tx2_60.base_mesh.T.copy()
    configurations = test_configurations(tx2_60)
    for name, q in configurations:
        if args.headless:
            tx2_60.robot.q = q
            env.step(0.02, render=False)
        else:
            animate_to(env, tx2_60, q)
        verify_configuration(env, tx2_60, name, q, local_vertices)
        if not np.allclose(tx2_60.base_mesh.T, base_transform, atol=1e-8):
            raise AssertionError(f"{name}: base mesh rotated or translated with q1")

    print(
        f"Passed {len(configurations)} configurations; "
        "base mesh remained static and all link meshes followed their DH frames."
    )
    if not args.headless:
        print("Swift is displaying the final configuration; press Ctrl+C to close.")
        while True:
            env.step(0.05)


if __name__ == "__main__":
    main()
