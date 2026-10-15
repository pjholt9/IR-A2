"""Kinematic recording and plots for EC66 task execution.

The worst-case Jacobian metrics exclude standby and transfers, since the
initial standby pose may be singular. This is not collision validation.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

    valid_indices = np.where(np.isin(phases, ["Approach", "Door opening"]))[0]



    if len(valid_indices) == 0:
        raise ValueError("No approach or door-opening samples recorded")

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



