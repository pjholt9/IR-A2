

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from spatialmath.base import tr2delta

if __package__:
    from ..collision import in_collision
    from ..tx2_60 import TX2_60
else:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from collision import in_collision
    from tx2_60 import TX2_60


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_output"
START_Q = np.array([0.2, -1.0, -0.8, 0.2, -0.7, 0.5])
TEST_DISPLACEMENT = np.array([0.025, 0.020, -0.015])


@dataclass
class RMRCResult:
    times: np.ndarray
    joint_positions: np.ndarray
    desired_positions: np.ndarray
    actual_positions: np.ndarray
    position_errors: np.ndarray
    minimum_singular_values: np.ndarray
    damping_values: np.ndarray


def quintic_blend(progress):
    """Return a zero-velocity, zero-acceleration quintic path blend and rate."""
    progress = float(np.clip(progress, 0.0, 1.0))
    blend = 10 * progress**3 - 15 * progress**4 + 6 * progress**5
    rate = 30 * progress**2 - 60 * progress**3 + 30 * progress**4
    return blend, rate


def run_rmrc(
    tx2_60,
    start_q=START_Q,
    displacement=TEST_DISPLACEMENT,
    duration=4.0,
    dt=0.02,
    proportional_gain=5.0,
    minimum_damping=0.01,
    damping_threshold=0.08,
    damping_gain=1.2,
    max_joint_speed=1.0,
    env=None,
):
    """Track a Cartesian translation using adaptive damped least squares."""
    if duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must both be positive")
    if proportional_gain <= 0 or minimum_damping <= 0:
        raise ValueError("controller gain and minimum damping must be positive")
    if max_joint_speed <= 0:
        raise ValueError("max_joint_speed must be positive")

    robot = tx2_60.robot
    q = np.asarray(start_q, dtype=float).copy()
    displacement = np.asarray(displacement, dtype=float)
    if q.shape != (robot.n,) or displacement.shape != (3,):
        raise ValueError("start_q must match robot.n and displacement must be xyz")

    qlim = np.asarray(robot.qlim)
    if np.any(q < qlim[0]) or np.any(q > qlim[1]):
        raise ValueError("start_q is outside the robot's joint limits")
    robot.q = q
    initial_pose = robot.fkine(q)
    initial_position = initial_pose.t.copy()

    step_count = int(np.ceil(duration / dt))
    times = np.empty(step_count + 1)
    times[0] = 0.0
    joint_positions = np.empty((step_count + 1, robot.n))
    joint_positions[0] = q
    desired_positions = np.empty((step_count + 1, 3))
    desired_positions[0] = initial_position
    actual_positions = np.empty((step_count + 1, 3))
    actual_positions[0] = initial_position
    position_errors = np.empty(step_count + 1)
    position_errors[0] = 0.0
    minimum_singular_values = np.empty(step_count)
    damping_values = np.empty(step_count)

    for index in range(step_count):
        elapsed = min((index + 1) * dt, duration)
        step_dt = elapsed - times[index]
        # Desired position/velocity along a smooth start-to-goal line
        blend, blend_rate = quintic_blend(elapsed / duration)
        desired_position = initial_position + blend * displacement
        desired_velocity = (blend_rate / duration) * displacement

        # Task-space velocity = feedforward path velocity + P correction of position error,
        # plus a P term that holds the starting orientation (angular error from tr2delta)
        current_pose = robot.fkine(q)
        pose_error = tr2delta(current_pose.A, initial_pose.A)
        task_velocity = np.concatenate(
            (
                desired_velocity
                + proportional_gain * (desired_position - current_pose.t),
                proportional_gain * pose_error[3:],
            )
        )
        jacobian = robot.jacob0(q)
        # Smallest singular value measures closeness to a singularity (0 = singular)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        sigma_min = float(singular_values[-1])
        # Adaptive damping: stays at minimum_damping while sigma_min >= damping_threshold,
        # then grows linearly as sigma_min falls below it, trading accuracy for bounded joint speeds
        damping = minimum_damping + damping_gain * max(
            0.0, damping_threshold - sigma_min
        )
        # Damped least squares: qdot = J^T (J J^T + damping^2 I)^-1 v
        joint_velocity = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(robot.n),
            task_velocity,
        )
        joint_velocity = np.clip(
            joint_velocity, -max_joint_speed, max_joint_speed
        )
        # Stop any joint already at a limit from moving further past it
        joint_velocity[(q <= qlim[0]) & (joint_velocity < 0)] = 0.0
        joint_velocity[(q >= qlim[1]) & (joint_velocity > 0)] = 0.0
        # Euler-integrate to the next joint configuration
        q_next = q + step_dt * joint_velocity
        if np.any(q_next < qlim[0] - 1e-10) or np.any(
            q_next > qlim[1] + 1e-10
        ):
            raise RuntimeError("RMRC step would exceed a joint limit")

        # Collision avoidance: hold still in Swift until clear, or fail when headless
        if tx2_60.obstacles:
            if env is None:
                if in_collision(robot, q_next, tx2_60.obstacles):
                    raise RuntimeError(
                        "RMRC step is blocked; provide a Swift environment "
                        "to use collision waiting."
                    )
            else:
                tx2_60._wait_until_clear(env, q_next)

        q = q_next
        robot.q = q
        if env is not None:
            env.step(step_dt)
            for hook in tx2_60.step_hooks:
                hook(step_dt)

        # Log this step for plotting
        actual_position = robot.fkine(q).t.copy()
        times[index + 1] = elapsed
        joint_positions[index + 1] = q
        desired_positions[index + 1] = desired_position
        actual_positions[index + 1] = actual_position
        position_errors[index + 1] = np.linalg.norm(
            desired_position - actual_position
        )
        minimum_singular_values[index] = sigma_min
        damping_values[index] = damping

    return RMRCResult(
        times=times,
        joint_positions=joint_positions,
        desired_positions=desired_positions,
        actual_positions=actual_positions,
        position_errors=position_errors,
        minimum_singular_values=minimum_singular_values,
        damping_values=damping_values,
    )


def plot_rmrc(result, output_dir):
    """Save Cartesian tracking, conditioning, damping, and joint plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(15, 5))
    path_axis = figure.add_subplot(131, projection="3d")
    path_axis.plot(
        result.desired_positions[:, 0],
        result.desired_positions[:, 1],
        result.desired_positions[:, 2],
        "--",
        label="Desired",
    )
    path_axis.plot(
        result.actual_positions[:, 0],
        result.actual_positions[:, 1],
        result.actual_positions[:, 2],
        label="Actual",
    )
    path_axis.set_xlabel("X (m)")
    path_axis.set_ylabel("Y (m)")
    path_axis.set_zlabel("Z (m)")
    path_axis.set_title("Cartesian RMRC path")
    path_axis.legend()

    error_axis = figure.add_subplot(132)
    error_axis.plot(result.times, result.position_errors)
    error_axis.set_xlabel("Time (s)")
    error_axis.set_ylabel("Position error (m)")
    error_axis.set_title("Cartesian tracking error")
    error_axis.grid(True)

    conditioning_axis = figure.add_subplot(133)
    sample_times = result.times[1:]
    conditioning_axis.plot(
        sample_times,
        result.minimum_singular_values,
        label="Minimum singular value",
    )
    conditioning_axis.plot(
        sample_times,
        result.damping_values,
        label="Damping",
    )
    conditioning_axis.set_xlabel("Time (s)")
    conditioning_axis.set_title("Singularity-aware damping")
    conditioning_axis.grid(True)
    conditioning_axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "tx2_60_rmrc_tracking.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    for joint in range(result.joint_positions.shape[1]):
        axis.plot(
            result.times,
            np.rad2deg(result.joint_positions[:, joint]),
            label=f"Joint {joint + 1}",
        )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Joint position (degrees)")
    axis.set_title("RMRC joint motion")
    axis.grid(True)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "tx2_60_rmrc_joints.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    tx2_60 = TX2_60()
    robot = tx2_60.create_robot()
    env = None
    if args.show:
        from spatialgeometry import Axes
        from swift import Swift

        robot.control_mode = "p"
        for link in robot.links:
            link.geometry = [*link.geometry, Axes(0.06)]
        env = Swift()
        env.launch(realtime=True)
        tx2_60.create_mesh_robot(env)
        env.add(Axes(0.1, pose=tx2_60.base))

    result = run_rmrc(tx2_60, env=env)
    final_error = float(result.position_errors[-1])
    if final_error > 0.002:
        raise AssertionError(
            f"RMRC final Cartesian error {final_error:.4f} m exceeds 0.002 m"
        )
    qlim = np.asarray(robot.qlim)
    if np.any(result.joint_positions < qlim[0] - 1e-10) or np.any(
        result.joint_positions > qlim[1] + 1e-10
    ):
        raise AssertionError("RMRC test exceeded a joint limit")

    plot_rmrc(result, args.output_dir)
    print(
        f"RMRC passed: {len(result.times) - 1} steps, "
        f"final position error={final_error:.6f} m, "
        f"peak error={result.position_errors.max():.6f} m, "
        f"minimum sigma={result.minimum_singular_values.min():.6g}."
    )
    print(f"Plots saved in: {args.output_dir.resolve()}")

    if env is not None:
        print("Swift is displaying the final robot pose; press Ctrl+C to close.")
        while True:
            env.step(0.05)


if __name__ == "__main__":
    main()
