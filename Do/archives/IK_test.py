
import swift
import numpy as np
from spatialmath import SE3
from math import pi
from scipy import linalg
from scenery import Scenery
from roboticstoolbox import jtraj
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from spatialmath.base import tr2angvec
from scipy.spatial.transform import Rotation

class KinematicRecorder:
    def __init__(self, robot):
        self.robot = robot
        self.time = [0.0]
        self.q = [robot.q.copy()]
        self.phase = ["Standby"]

    def record(self, dt, phase):
        self.time.append(self.time[-1] + dt)
        self.q.append(self.robot.q.copy())
        self.phase.append(phase)
        
def analyse_full_kinematics(robot, recorder):

    time = np.array(recorder.time)
    q = np.array(recorder.q)
    phases = np.array(recorder.phase)

    q_deg = np.rad2deg(q)

    # Joint velocities: different dt for each phase
    dt = np.diff(time)
    qdot = np.diff(q, axis=0) / dt[:, None]
    qdot_deg = np.rad2deg(qdot)
    velocity_time = (time[:-1] + time[1:]) / 2

    # Jacobian metrics
    manipulability = []
    sigma_min = []

    for qi in q:
        J = robot.jacob0(qi)

        manipulability.append(
            robot.manipulability(qi)
        )

        sigma_min.append(
            np.linalg.svd(J, compute_uv=False)[-1]
        )

    manipulability = np.array(manipulability)
    sigma_min = np.array(sigma_min)

    qlim = np.array([link.qlim for link in robot.links])

    # Phase boundaries
    boundaries = []

    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            boundaries.append((time[i], phases[i]))

    def add_phase_lines(ax):
        for t, phase in boundaries:
            ax.axvline(t, color="gray", linestyle="--",
                       linewidth=1, alpha=0.6)

    # ---------------------------------------------- #
    # Figure 1: Joint positions
    # ---------------------------------------------- #

    fig1, axes = plt.subplots(
        3, 2, figsize=(12, 9), sharex=True
    )

    fig1.suptitle("EC66 - Full Task Joint Positions")

    for j, ax in enumerate(axes.flat):

        ax.plot(time, q_deg[:, j])

        ax.axhline(
            np.rad2deg(qlim[j, 0]),
            color="red", linestyle="--"
        )

        ax.axhline(
            np.rad2deg(qlim[j, 1]),
            color="red", linestyle="--"
        )

        add_phase_lines(ax)

        ax.set_title(f"Joint {j + 1}")
        ax.set_ylabel("Angle (deg)")
        ax.grid(True)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")

    fig1.tight_layout()

    # ---------------------------------------------- #
    # Figure 2: Joint velocities
    # ---------------------------------------------- #

    fig2, axes = plt.subplots(
        3, 2, figsize=(12, 9), sharex=True
    )

    fig2.suptitle("EC66 - Full Task Joint Velocities")

    for j, ax in enumerate(axes.flat):

        ax.plot(velocity_time, qdot_deg[:, j])

        add_phase_lines(ax)

        ax.set_title(f"Joint {j + 1}")
        ax.set_ylabel("Velocity (deg/s)")
        ax.grid(True)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")

    fig2.tight_layout()

    # ---------------------------------------------- #
    # Figure 3: Kinematic performance
    # ---------------------------------------------- #

    fig3, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True
    )

    fig3.suptitle("EC66 - Full Task Kinematic Performance")

    axes[0].plot(time, manipulability)
    axes[0].set_ylabel("Manipulability")

    axes[1].plot(time, sigma_min)
    axes[1].set_ylabel("Minimum singular value")
    axes[1].set_xlabel("Time (s)")

    for ax in axes:
        add_phase_lines(ax)
        ax.grid(True)

    fig3.tight_layout()

    # ---------------------------------------------- #
    # Figure 4: Animated velocity ellipsoid
    # ---------------------------------------------- #

    # Precompute ellipsoid geometry for efficiency
    u = np.linspace(0, 2*np.pi, 24)
    v = np.linspace(0, np.pi, 16)

    sphere = np.array([
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v))
    ])

    sphere_flat = sphere.reshape(3, -1)

    # Every fourth sample for smoother plotting performance
    frame_indices = np.unique(
        np.r_[np.arange(0, len(q), 4), len(q) - 1]
    )

    ellipsoids = []
    global_radius = 0.0

    for idx in frame_indices:

        Jv = robot.jacob0(q[idx])[:3, :]

        U, S, _ = np.linalg.svd(
            Jv, full_matrices=False
        )

        surface = (
            U @ np.diag(S) @ sphere_flat
        ).reshape(sphere.shape)

        ellipsoids.append(surface)

        global_radius = max(global_radius, S.max())

    fig4 = plt.figure(figsize=(9, 8))
    ax = fig4.add_subplot(111, projection="3d")

    def update(frame):

        ax.clear()

        idx = frame_indices[frame]
        surface = ellipsoids[frame]

        ax.plot_surface(
            surface[0],
            surface[1],
            surface[2],
            alpha=0.55,
            color="steelblue",
            edgecolor="none"
        )

        ax.set_xlim(-global_radius, global_radius)
        ax.set_ylim(-global_radius, global_radius)
        ax.set_zlim(-global_radius, global_radius)

        ax.set_box_aspect((1, 1, 1))

        ax.set_xlabel("Vx (m/s)")
        ax.set_ylabel("Vy (m/s)")
        ax.set_zlabel("Vz (m/s)")

        ax.set_title(
            f"EC66 Translational Velocity Ellipsoid\n"
            f"Phase: {phases[idx]} | Time: {time[idx]:.2f} s"
        )

        return ()

    animation = FuncAnimation(
        fig4,
        update,
        frames=len(frame_indices),
        interval=50,
        repeat=True
    )

    # ---------------------------------------------- #
    # Summary
    # ---------------------------------------------- #

    # Exclude the initial standby sample from worst-case analysis
    valid_indices = np.where(phases != "Standby")[0]

    worst_idx = valid_indices[
        np.argmin(sigma_min[valid_indices])
    ]

    min_manip_idx = valid_indices[
        np.argmin(manipulability[valid_indices])
    ]

    print("\n========== FULL KINEMATIC ANALYSIS ==========")

    print(f"Total samples: {len(q)}")
    print(f"Duration: {time[-1]:.2f} s")

    print(
        f"Maximum joint speed: "
        f"{np.max(np.abs(qdot_deg)):.2f} deg/s"
    )

    print(
        f"Minimum manipulability (excluding standby): "
        f"{manipulability[min_manip_idx]:.6f}"
    )
    
    print(
        f"Minimum singular value (excluding standby): "
        f"{sigma_min[worst_idx]:.6f}"
    )
    
    print(f"Worst phase: {phases[worst_idx]}")
    print(f"Worst time: {time[worst_idx]:.2f} s")

    print(
        "Worst joint configuration (deg):",
        np.round(q_deg[worst_idx], 2)
    )

    violations = np.any(
        (q < qlim[:, 0]) | (q > qlim[:, 1])
    )

    print(f"Joint limit violations: {violations}")

    print("============================================\n")

    plt.show()

    return animation

def analyse_door_kinematics(robot, q_trajectory, door_angles, dt=0.05):
    """
    Kinematic analysis of the EC66 door-opening trajectory.

    Parameters
    ----------
    robot : DHRobot
        EC66 robot model.
    q_trajectory : array_like, shape (N, 6)
        Joint configurations along the door-opening trajectory.
    door_angles : array_like, shape (N,)
        Corresponding door angles in radians.
    dt : float
        Assumed time interval between trajectory samples.
    """

    q = np.asarray(q_trajectory, dtype=float)
    angles = np.asarray(door_angles, dtype=float)

    if q.ndim != 2 or q.shape[1] != robot.n:
        raise ValueError("q_trajectory must have shape (N, robot.n)")

    if len(q) != len(angles) or len(q) < 2:
        raise ValueError("Trajectory and door angles must have matching lengths >= 2")

    if dt <= 0:
        raise ValueError("dt must be positive")

    time = np.arange(len(q)) * dt
    door_deg = np.rad2deg(angles)

    # -------------------------------------------------- #
    # Joint positions and velocities
    # -------------------------------------------------- #

    q_deg = np.rad2deg(q)

    # Velocities are defined between consecutive samples
    qdot = np.diff(q, axis=0) / dt
    qdot_deg = np.rad2deg(qdot)

    velocity_time = (time[:-1] + time[1:]) / 2

    qlim = np.array([
        link.qlim for link in robot.links
    ])

    # -------------------------------------------------- #
    # Jacobian-based kinematic metrics
    # -------------------------------------------------- #

    manipulability = []
    sigma_min = []

    for qi in q:
        J = robot.jacob0(qi)

        manipulability.append(
            robot.manipulability(qi)
        )

        singular_values = np.linalg.svd(
            J, compute_uv=False
        )

        sigma_min.append(
            np.min(singular_values)
        )

    manipulability = np.array(manipulability)
    sigma_min = np.array(sigma_min)

    # -------------------------------------------------- #
    # Figure 1: Joint positions
    # -------------------------------------------------- #

    fig1, axes1 = plt.subplots(
        3, 2, figsize=(12, 9), sharex=True
    )

    fig1.suptitle(
        "EC66 Door Opening - Joint Positions",
        fontsize=14
    )

    for j, ax in enumerate(axes1.flat):
        ax.plot(
            door_deg,
            q_deg[:, j],
            label=f"Joint {j + 1}"
        )

        ax.axhline(
            np.rad2deg(qlim[j, 0]),
            color="red",
            linestyle="--",
            linewidth=1,
            label="Joint limits"
        )

        ax.axhline(
            np.rad2deg(qlim[j, 1]),
            color="red",
            linestyle="--",
            linewidth=1
        )

        ax.set_ylabel("Angle (deg)")
        ax.set_title(f"Joint {j + 1}")
        ax.grid(True)
        ax.legend(loc="best")

    for ax in axes1[-1, :]:
        ax.set_xlabel("Door angle (deg)")

    fig1.tight_layout()

    # -------------------------------------------------- #
    # Figure 2: Joint velocities
    # -------------------------------------------------- #

    fig2, axes2 = plt.subplots(
        3, 2, figsize=(12, 9), sharex=True
    )

    fig2.suptitle(
        "EC66 Door Opening - Joint Velocities",
        fontsize=14
    )

    for j, ax in enumerate(axes2.flat):
        ax.plot(
            velocity_time,
            qdot_deg[:, j]
        )

        ax.set_ylabel("Velocity (deg/s)")
        ax.set_title(f"Joint {j + 1}")
        ax.grid(True)

    for ax in axes2[-1, :]:
        ax.set_xlabel("Time (s)")

    fig2.tight_layout()

    # -------------------------------------------------- #
    # Figure 3: Manipulability and singular values
    # -------------------------------------------------- #

    fig3, axes3 = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True
    )

    fig3.suptitle(
        "EC66 Door Opening - Kinematic Performance",
        fontsize=14
    )

    axes3[0].plot(
        door_deg,
        manipulability,
        color="blue"
    )

    axes3[0].set_ylabel("Manipulability")
    axes3[0].grid(True)

    axes3[1].plot(
        door_deg,
        sigma_min,
        color="purple"
    )

    axes3[1].set_ylabel("Minimum singular value")
    axes3[1].set_xlabel("Door angle (deg)")
    axes3[1].grid(True)

    fig3.tight_layout()

    # -------------------------------------------------- #
    # Figure 4: Translational velocity ellipsoids
    # -------------------------------------------------- #

    sample_angles = [0, -pi/4, -pi/2]

    fig4 = plt.figure(figsize=(15, 5))
    fig4.suptitle(
        "EC66 Translational Velocity Ellipsoids",
        fontsize=14
    )

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 25)

    sphere = np.array([
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v))
    ])

    for k, target_angle in enumerate(sample_angles):

        index = np.argmin(
            np.abs(angles - target_angle)
        )

        Jv = robot.jacob0(q[index])[:3, :]

        U, S, _ = np.linalg.svd(
            Jv,
            full_matrices=False
        )

        # Velocity ellipsoid:
        # xdot = Jv @ qdot, with ||qdot|| <= 1
        ellipsoid = (
            U @ np.diag(S) @ sphere.reshape(3, -1)
        ).reshape(3, *sphere.shape[1:])

        ax = fig4.add_subplot(
            1, 3, k + 1, projection="3d"
        )

        ax.plot_surface(
            ellipsoid[0],
            ellipsoid[1],
            ellipsoid[2],
            alpha=0.45,
            edgecolor="none"
        )

        ax.set_title(
            f"Door angle: {np.rad2deg(angles[index]):.0f}°"
        )

        ax.set_xlabel("Vx")
        ax.set_ylabel("Vy")
        ax.set_zlabel("Vz")

        # Same scale on all axes, so shapes are not distorted
        max_radius = max(S.max(), 1e-9)

        ax.set_xlim(-max_radius, max_radius)
        ax.set_ylim(-max_radius, max_radius)
        ax.set_zlim(-max_radius, max_radius)

        ax.set_box_aspect((1, 1, 1))

    fig4.tight_layout()

    # -------------------------------------------------- #
    # Summary
    # -------------------------------------------------- #

    joint_limit_violations = np.any(
        (q < qlim[:, 0]) | (q > qlim[:, 1])
    )

    print("\n========== KINEMATIC ANALYSIS ==========")
    print(f"Number of waypoints: {len(q)}")
    print(f"Assumed trajectory duration: {time[-1]:.2f} s")
    print(f"Maximum joint speed: {np.max(np.abs(qdot_deg)):.2f} deg/s")
    print(f"Minimum manipulability: {manipulability.min():.6f}")
    print(f"Minimum singular value: {sigma_min.min():.6f}")
    print(f"Joint limit violations: {joint_limit_violations}")
    print("========================================\n")
    
    print("\n========== PERFORMANCE BY PHASE ==========")

    for phase_name in ["Pre-grasp", "Approach", "Door opening"]:

        mask = phases == phase_name

        if not np.any(mask):
            continue

        print(f"\n{phase_name}:")

        print(
            f"  Minimum manipulability: "
            f"{manipulability[mask].min():.6f}"
        )

        print(
            f"  Minimum singular value: "
            f"{sigma_min[mask].min():.6f}"
        )

        velocity_mask = phases[1:] == phase_name
        max_speed = np.max(np.abs(qdot_deg[velocity_mask]))

        print(f"  Maximum joint speed: {max_speed:.2f} deg/s")

    plt.show()

    return {
        "q": q,
        "qdot": qdot,
        "manipulability": manipulability,
        "sigma_min": sigma_min,
        "joint_limit_violations": joint_limit_violations
    }


env = swift.Swift()
env.launch(realtime=True)

scenery = Scenery()
scenery.add_to_env(env)

env.step(1)

# -------------------------------------------------- #
# Move platform to Station
# -------------------------------------------------- #

target_printer = scenery.printer2
scenery.move_to_printer(target_printer, env)

robot = scenery.robot
door = scenery.printer2.door

# Start from standby
robot.q = scenery.ec66.q_standby.copy()
door.update_angle(0)

env.step(1)

recorder = KinematicRecorder(robot)


# -------------------------------------------------- #
# Phase 1: Standby -> Pre-grasp (IK + jtraj)
# -------------------------------------------------- #

T_handle = door.get_handle_pose()

T_contact = T_handle @ SE3.Rx(-pi/2)

# 10 cm away from handle along EE local Z
T_pregrasp = T_contact @ SE3(0, 0, -0.10)

solution = robot.ikine_LM(
    T_pregrasp,
    q0=robot.q,
    joint_limits=True
)

print("Pre-grasp IK success:", solution.success)

if not solution.success:
    raise RuntimeError("Pre-grasp IK failed")

trajectory = jtraj(robot.q.copy(), solution.q, 200)

for q in trajectory.q:
    robot.q = q
    env.step(0.02)
    
    recorder.record(0.02, "Pre-grasp")

print("Reached pre-grasp")
env.step(0.5)


# -------------------------------------------------- #
# Phase 2: Pre-grasp -> Handle (6-DoF Closed-loop RMRC)
# -------------------------------------------------- #

steps = 150
delta_t = 0.03
damping = 0.001
Kp = 2.0
KR = 2.0

start_pos = robot.fkine(robot.q).t.copy()
end_pos = T_contact.t.copy()

positions = np.linspace(start_pos, end_pos, steps)

for i in range(steps - 1):

    # Current EE pose
    T_actual = robot.fkine(robot.q)

    # ---------------------------------------------- #
    # Linear velocity (XYZ)
    # ---------------------------------------------- #

    x_desired = positions[i + 1]

    xdot_desired = (
        positions[i + 1] - positions[i]
    ) / delta_t

    position_error = x_desired - T_actual.t

    linear_velocity = (
        xdot_desired + Kp * position_error
    )

    # ---------------------------------------------- #
    # Angular velocity (orientation control)
    # ---------------------------------------------- #

    R_desired = T_contact.R
    R_actual = T_actual.R

    R_error = R_desired @ R_actual.T

    rotation_error = Rotation.from_matrix(
        R_error
    ).as_rotvec()

    angular_velocity = KR * rotation_error

    # ---------------------------------------------- #
    # Full 6-DoF Cartesian velocity
    # ---------------------------------------------- #

    xdot = np.r_[
        linear_velocity,
        angular_velocity
    ]

    J = robot.jacob0(robot.q)

    # Dimension checks
    assert J.shape == (6, 6)
    assert xdot.shape == (6,)

    # Damped Least Squares
    qdot = J.T @ linalg.solve(
        J @ J.T + damping * np.eye(6),
        xdot
    )

    q_next = robot.q + delta_t * qdot

    # ---------------------------------------------- #
    # Joint limit check
    # ---------------------------------------------- #

    within_limits = all(
        robot.links[j].qlim[0] <= q_next[j] <= robot.links[j].qlim[1]
        for j in range(robot.n)
    )

    if not within_limits:
        raise RuntimeError(
            "RMRC stopped: joint limit reached"
        )

    robot.q = q_next

    env.step(delta_t)
    recorder.record(delta_t, "Approach")

# -------------------------------------------------- #
# Check final position and orientation errors
# -------------------------------------------------- #

T_actual = robot.fkine(robot.q)

position_error = np.linalg.norm(
    T_contact.t - T_actual.t
)

orientation_error = np.linalg.norm(
    Rotation.from_matrix(
        T_contact.R @ T_actual.R.T
    ).as_rotvec()
)

print("\n========== END OF APPROACH ==========")
print("Position error (mm):", position_error * 1000)
print("Orientation error (deg):", np.rad2deg(orientation_error))
print("====================================\n")

if position_error > 0.005:
    raise RuntimeError(
        "EE did not reach handle accurately"
    )

if orientation_error > np.deg2rad(2):
    raise RuntimeError(
        "EE orientation error exceeds 2 degrees"
    )

print("Reached handle")
env.step(0.5)
# -------------------------------------------------- #
# Phase 3: Open door (Continuous IK trajectory)
# -------------------------------------------------- #

angles = np.linspace(0, -pi/2, 91)

q_solutions = []
q_guess = robot.q.copy()

T_actual = robot.fkine(robot.q)

R_error = T_contact.R @ T_actual.R.T

angle_error, axis = tr2angvec(R_error)

print("\n========== END OF APPROACH ==========")
print(
    "Position error (mm):",
    np.linalg.norm(T_contact.t - T_actual.t) * 1000
)
print(
    "Orientation error (deg):",
    np.rad2deg(abs(angle_error))
)
print("====================================\n")


# Calculate entire door-opening trajectory first
for angle in angles:

    door.update_angle(angle)

    T_handle = door.get_handle_pose()
    T_target = T_handle @ SE3.Rx(-pi/2)

    solution = robot.ikine_LM(
        T_target,
        q0=q_guess,
        joint_limits=True
    )

    if not solution.success:
        door.update_angle(0)
        raise RuntimeError(
            f"Door IK failed at {np.rad2deg(angle):.1f} degrees"
        )

    q_solutions.append(solution.q.copy())
    q_guess = solution.q.copy()

# Reset door before animation
door.update_angle(0)

# Check joint continuity
q_array = np.array(q_solutions)

# Check continuity between RMRC approach and door IK
joint_jump = np.rad2deg(
    q_array[0] - recorder.q[-1]
)

print(
    "Phase 2 -> Phase 3 joint jump (deg):",
    np.round(joint_jump, 3)
)

if np.max(np.abs(joint_jump)) > 2:
    raise RuntimeError(
        "Large joint jump between approach and door opening"
    )

joint_changes = np.abs(np.diff(q_array, axis=0))
max_change = np.rad2deg(np.max(joint_changes))  

print("Maximum joint change per step:", max_change, "deg")

if max_change > 10:
    raise RuntimeError("Large joint jump detected in door trajectory")



# Animate door and robot together
for i in range(len(angles)):

    door.update_angle(angles[i])
    robot.q = q_solutions[i].copy()

    env.step(0.05)
    
    recorder.record(0.05, "Door opening")

print("Door fully opened!")
env.step(1)



# -------------------------------------------------- #
# Kinematic analysis of Printer 1 door opening
# -------------------------------------------------- #

ellipsoid_animation = analyse_full_kinematics(
    robot,
    recorder
)


# -------------------------------------------------- #
# Keep Swift running
# -------------------------------------------------- #

while True:
    env.step(0.05)
