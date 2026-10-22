"""Sample TX2-60 reachability, singularities, IK, and joint trajectories.

Run from the repository root with:
    python -m PH.tx2_60_kinematic_analysis
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from roboticstoolbox import jtraj
from spatialmath.base import tr2delta

if __package__:
    from ..tx2_60 import TX2_60
else:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tx2_60 import TX2_60


# Fixed test poses keep comparisons repeatable; the random sample seed can still be set at run time.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_output"
IK_TEST_Q = np.array([0.2, -1.0, -0.8, 0.2, -0.7, 0.5])
TRAJECTORY_END_Q = np.array([0.3, -0.9, -0.9, 0.35, -0.6, 0.4])


# Sample legal joint angles to see the reachable region and how well-conditioned each pose is.
def sample_workspace(tx2_60, sample_count=1200, seed=23):
    """Sample end-effector positions and translational manipulability."""
    robot = tx2_60.robot
    qlim = np.asarray(robot.qlim)
    rng = np.random.default_rng(seed)
    configurations = rng.uniform(qlim[0], qlim[1], size=(sample_count, robot.n))
    positions = np.empty((sample_count, 3))
    manipulability = np.empty(sample_count)
    minimum_singular_values = np.empty(sample_count)

    for index, q in enumerate(configurations):
        positions[index] = robot.fkine(q).t
        jacobian = robot.jacob0(q)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        minimum_singular_values[index] = singular_values[-1]  # A small value flags a nearly singular pose.
        # Yoshikawa manipulability of the position rows only: sqrt(det(Jv Jv^T))
        translational_jacobian = jacobian[:3, :]
        eigenvalues = np.linalg.eigvalsh(
            translational_jacobian @ translational_jacobian.T
        )
        manipulability[index] = np.sqrt(np.prod(np.maximum(eigenvalues, 0.0)))  # Clip tiny negative round-off before taking the square root.

    return positions, manipulability, minimum_singular_values


# Try IK on a pose known to be reachable, then check a smooth joint-space move stays within limits.
def verify_ik_and_joint_trajectory(tx2_60):
    """Check FK/IK recovery and joint-trajectory endpoint and limit behavior."""
    robot = tx2_60.robot
    # FK gives a pose known to be reachable; IK starts from a perturbed guess and must recover it
    target_pose = robot.fkine(IK_TEST_Q)
    q0 = IK_TEST_Q + np.array([0.02, -0.02, 0.02, -0.02, 0.02, -0.02])
    solution = robot.ikine_LM(target_pose, q0=q0)
    if not solution.success:
        raise AssertionError("IK failed to recover a known reachable FK pose.")

    recovered_pose = robot.fkine(solution.q)
    position_error = float(np.linalg.norm(recovered_pose.t - target_pose.t))
    # tr2delta puts the small-angle orientation error after the three translation components.
    rotation_error = float(
        np.linalg.norm(tr2delta(recovered_pose.A, target_pose.A)[3:])
    )
    if position_error > 1e-5 or rotation_error > 1e-4:
        raise AssertionError(
            "IK round-trip exceeded tolerance: "
            f"position={position_error:.3g} m, "
            f"orientation={rotation_error:.3g} rad."
        )

    # Joint-space trajectory must start/end on the requested configurations and respect qlim
    trajectory = jtraj(IK_TEST_Q, TRAJECTORY_END_Q, 100).q
    if not np.allclose(trajectory[0], IK_TEST_Q, atol=1e-10):
        raise AssertionError("Joint trajectory does not start at the requested q.")
    if not np.allclose(trajectory[-1], TRAJECTORY_END_Q, atol=1e-10):
        raise AssertionError("Joint trajectory does not end at the requested q.")
    qlim = np.asarray(robot.qlim)
    if np.any(trajectory < qlim[0] - 1e-10) or np.any(
        trajectory > qlim[1] + 1e-10
    ):
        raise AssertionError("Joint trajectory exceeds the robot's qlim.")

    return trajectory, position_error, rotation_error


# Plot the sampled workspace and the focused sweep that shows how the wrist singularity develops.
def plot_analysis(positions, manipulability, q5_values, minimum_singular_values, output_dir):
    """Save workspace/manipulability and singularity-sweep figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Colouring points by manipulability highlights reachable areas that are harder to move through.
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=manipulability,
        cmap="viridis",
        s=8,
        alpha=0.8,
    )
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title("TX2-60 sampled workspace and translational manipulability")
    figure.colorbar(scatter, ax=axis, label="Translational manipulability")
    figure.tight_layout()
    figure.savefig(output_dir / "tx2_60_workspace.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(np.rad2deg(q5_values), minimum_singular_values)
    axis.set_xlabel("Joint 5 (degrees)")
    axis.set_ylabel("Minimum singular value of Jacobian")
    axis.set_title("TX2-60 wrist singularity sweep")
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(output_dir / "tx2_60_singularity_sweep.png", dpi=160)
    plt.close(figure)


# Bring together the random samples, FK/IK checks, trajectory check, and joint-5 singularity sweep.
def analyze(tx2_60, sample_count=1200, seed=23, output_dir=DEFAULT_OUTPUT_DIR):
    """Run numerical checks, save plots, and return the sampled data."""
    positions, manipulability, minimum_singular_values = sample_workspace(
        tx2_60, sample_count=sample_count, seed=seed
    )
    trajectory, position_error, rotation_error = verify_ik_and_joint_trajectory(
        tx2_60
    )

    # Singularity sweep: vary only joint 5 and record sigma_min; it drops to 0 at q5 = 0
    # (wrist singularity, joints 4 and 6 axes align)
    q5_values = np.linspace(-np.pi / 2, np.pi / 2, 121)
    sweep_q = IK_TEST_Q.copy()
    sweep_singular_values = np.empty(q5_values.size)
    for index, q5 in enumerate(q5_values):
        sweep_q[4] = q5
        sweep_singular_values[index] = np.linalg.svd(
            tx2_60.robot.jacob0(sweep_q), compute_uv=False
        )[-1]

    plot_analysis(
        positions,
        manipulability,
        q5_values,
        sweep_singular_values,
        output_dir,
    )

    print(f"Sampled configurations: {sample_count}")
    print(
        "Sampled XYZ bounds (m): "
        f"min={np.round(positions.min(axis=0), 3).tolist()}, "
        f"max={np.round(positions.max(axis=0), 3).tolist()}"
    )
    print(
        "Minimum 6D Jacobian singular value across samples: "
        f"{minimum_singular_values.min():.6g}"
    )
    print(
        f"IK round-trip errors: {position_error:.3g} m, "
        f"{rotation_error:.3g} rad"
    )
    print(
        "Joint trajectory: 100 samples, endpoints and joint limits verified."
    )
    print(f"Plots saved in: {output_dir.resolve()}")

    return {
        "positions": positions,
        "manipulability": manipulability,
        "minimum_singular_values": minimum_singular_values,
        "trajectory": trajectory,
        "q5_values": q5_values,
        "sweep_singular_values": sweep_singular_values,
    }


# Parse run options, build a mesh-free arm, and save the analysis plots to the chosen folder.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be a positive integer")

    tx2_60 = TX2_60(use_meshes=False)
    tx2_60.create_robot()
    analyze(
        tx2_60,
        sample_count=args.samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
