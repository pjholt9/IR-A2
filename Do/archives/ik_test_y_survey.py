"""EC66 single-station Y-offset survey.

Set station2 to X=0.80 m in linear_platform.py.
Change robot_offset Y in scenery.py between runs.
Pre-pre-grasp offset is expressed in WORLD coordinates relative to pre-grasp.
This simulation does not perform collision detection.
"""

from math import pi

import matplotlib.pyplot as plt
import numpy as np
import swift
from roboticstoolbox import jtraj
from scipy import linalg
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from scenery import Scenery


# ---------- Experiment parameters ----------
PRE_PREGRASP_OFFSET = np.array([-0.15, -0.03, 0.15])  # world XYZ, m
JOINT_STEP_LIMIT_DEG = 10.0


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


def check_limits(robot, q, description):
    limits = np.array([link.qlim for link in robot.links])
    if np.any(q < limits[:, 0] - 1e-9) or np.any(q > limits[:, 1] + 1e-9):
        raise RuntimeError(f"Joint limits exceeded: {description}")


def solve_ik(robot, target, seed, description):
    result = robot.ikine_LM(target, q0=seed.copy(), joint_limits=True)
    if not result.success:
        raise RuntimeError(f"IK failed: {description} ({result.reason})")
    check_limits(robot, result.q, description)
    return result.q.copy()


def animate_joint_move(robot, env, recorder, target_q, phase, steps=150, dt=0.02):
    trajectory = jtraj(robot.q.copy(), target_q, steps)
    # Skip duplicate starting configuration: time should only advance for real intervals.
    for q in trajectory.q[1:]:
        robot.q = q.copy()
        env.step(dt)
        recorder.record(dt, phase)


def approach_handle(robot, env, recorder, contact):
    """Existing 6D closed-loop RMRC controller (same gains as working code)."""
    steps, dt, damping, kp, kr = 150, 0.03, 0.001, 2.0, 2.0
    positions = np.linspace(robot.fkine(robot.q).t.copy(), contact.t.copy(), steps)

    for i in range(steps - 1):
        actual = robot.fkine(robot.q)
        linear = (positions[i + 1] - positions[i]) / dt
        linear += kp * (positions[i + 1] - actual.t)
        angular = kr * Rotation.from_matrix(contact.R @ actual.R.T).as_rotvec()
        twist = np.r_[linear, angular]
        J = robot.jacob0(robot.q)
        qdot = J.T @ linalg.solve(J @ J.T + damping * np.eye(6), twist)
        q_next = robot.q + dt * qdot
        check_limits(robot, q_next, "Approach RMRC")
        robot.q = q_next
        env.step(dt)
        recorder.record(dt, "Approach")

    actual = robot.fkine(robot.q)
    pos_error = np.linalg.norm(contact.t - actual.t)
    rot_error = np.linalg.norm(
        Rotation.from_matrix(contact.R @ actual.R.T).as_rotvec()
    )
    print(f"Handle error: {pos_error * 1000:.3f} mm, {np.rad2deg(rot_error):.4f} deg")
    if pos_error > 0.005 or rot_error > np.deg2rad(2):
        raise RuntimeError("Approach accuracy tolerance exceeded")


def open_door(robot, door, env, recorder):
    angles = np.linspace(0, -pi / 2, 91)
    q_waypoints = [robot.q.copy()]  # Exact start: no transition IK jump.
    seed = robot.q.copy()

    try:
        for angle in angles[1:]:
            door.update_angle(angle)
            target = door.get_handle_pose() @ SE3.Rx(-pi / 2)
            seed = solve_ik(robot, target, seed, f"Door {np.rad2deg(angle):.1f} deg")
            q_waypoints.append(seed)
    finally:
        door.update_angle(0)

    q_array = np.asarray(q_waypoints)
    max_step = np.rad2deg(np.abs(np.diff(q_array, axis=0))).max()
    print(f"Maximum door IK joint step: {max_step:.3f} deg")
    if max_step > JOINT_STEP_LIMIT_DEG:
        raise RuntimeError("Discontinuous door-opening IK trajectory")

    for angle, q in zip(angles[1:], q_waypoints[1:]):
        door.update_angle(angle)
        robot.q = q.copy()
        env.step(0.05)
        recorder.record(0.05, "Door opening")


def analyse_kinematics(robot, recorder, robot_y):
    time = np.asarray(recorder.time)
    q = np.asarray(recorder.q)
    phases = np.asarray(recorder.phase)
    q_deg = np.rad2deg(q)
    speed = np.rad2deg(np.diff(q, axis=0) / np.diff(time)[:, None])
    velocity_time = (time[:-1] + time[1:]) / 2

    J_all = [robot.jacob0(qi) for qi in q]
    sigma_min = np.array([np.linalg.svd(J, compute_uv=False)[-1] for J in J_all])
    manipulability = np.array([robot.manipulability(qi) for qi in q])
    qlim = np.array([link.qlim for link in robot.links])
    margins = np.rad2deg(np.minimum(q - qlim[:, 0], qlim[:, 1] - q))

    task = np.isin(phases, ["Approach", "Door opening"])
    if not np.any(task):
        raise RuntimeError("No manipulation-phase samples recorded")

    print("\n========== EC66 Y-OFFSET SURVEY ==========")
    print(f"Robot mounting offset Y: {robot_y:.3f} m")
    print("Rail station: station2 (set X=0.80 m in linear_platform.py)")
    print(f"Pre-pre-grasp offset: {PRE_PREGRASP_OFFSET} m (world)")
    print(f"Motion duration: {time[-1]:.2f} s")
    print("Metrics below: Approach + Door opening only")
    print(f"Minimum singular value (6D): {sigma_min[task].min():.6f}")
    print(f"Median manipulability (6D): {np.median(manipulability[task]):.6f}")
    print(f"Minimum manipulability (6D): {manipulability[task].min():.6f}")
    print(f"Minimum Joint 3 margin: {margins[task, 2].min():.2f} deg")
    print(f"Max joint speed (all phases): {np.abs(speed).max():.2f} deg/s")
    print(f"Joint limit violations: {np.any(margins < -1e-6)}")
    worst = np.where(task)[0][np.argmin(sigma_min[task])]
    print(f"Worst singular-value phase/time: {phases[worst]} / {time[worst]:.2f} s")
    print("Collision detection: NOT implemented")
    print("=========================================\n")

    boundaries = [time[i] for i in range(1, len(phases)) if phases[i] != phases[i - 1]]

    fig1, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    fig1.suptitle(f"EC66 joint angles | mounting Y={robot_y:.2f} m")
    for j, ax in enumerate(axes.flat):
        ax.plot(time, q_deg[:, j])
        ax.axhline(np.rad2deg(qlim[j, 0]), ls="--", color="red", lw=0.8)
        ax.axhline(np.rad2deg(qlim[j, 1]), ls="--", color="red", lw=0.8)
        for boundary in boundaries:
            ax.axvline(boundary, color="gray", ls=":", lw=0.8)
        ax.set_title(f"Joint {j + 1}")
        ax.set_ylabel("deg")
        ax.grid(True)
    axes[-1, 0].set_xlabel("Motion time (s)")
    axes[-1, 1].set_xlabel("Motion time (s)")
    fig1.tight_layout()

    fig2, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    fig2.suptitle(f"EC66 joint velocities | mounting Y={robot_y:.2f} m")
    for j, ax in enumerate(axes.flat):
        ax.plot(velocity_time, speed[:, j])
        for boundary in boundaries:
            ax.axvline(boundary, color="gray", ls=":", lw=0.8)
        ax.set_title(f"Joint {j + 1}")
        ax.set_ylabel("deg/s")
        ax.grid(True)
    axes[-1, 0].set_xlabel("Motion time (s)")
    axes[-1, 1].set_xlabel("Motion time (s)")
    fig2.tight_layout()

    fig3, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig3.suptitle(f"EC66 kinematics | mounting Y={robot_y:.2f} m")
    for ax, values, ylabel in zip(
        axes,
        [sigma_min, manipulability, margins[:, 2]],
        ["Minimum singular value (6D)", "Manipulability (6D)", "Joint 3 limit margin (deg)"],
    ):
        ax.plot(time, values)
        for boundary in boundaries:
            ax.axvline(boundary, color="gray", ls=":", lw=0.8)
        ax.set_ylabel(ylabel)
        ax.grid(True)
    axes[-1].set_xlabel("Motion time (s)")
    fig3.tight_layout()
    plt.show()


def main():
    env = swift.Swift()
    env.launch(realtime=True)
    scenery = Scenery()
    scenery.add_to_env(env)
    env.step(1)
    scenery.move2(env)

    robot = scenery.robot
    door = scenery.printer1.door
    robot.q = scenery.ec66.q_standby.copy()
    door.update_angle(0)
    env.step(0.5)

    robot_y = float(scenery.robot_offset.t[1])
    print(f"Robot mounting offset Y = {robot_y:.3f} m")
    recorder = KinematicRecorder(robot)

    contact = door.get_handle_pose() @ SE3.Rx(-pi / 2)
    pregrasp = contact @ SE3(0, 0, -0.10)
    pre_pregrasp = SE3.Rt(pregrasp.R, pregrasp.t + PRE_PREGRASP_OFFSET)

    q1 = solve_ik(robot, pre_pregrasp, robot.q, "Pre-pre-grasp")
    animate_joint_move(robot, env, recorder, q1, "Pre-pre-grasp")
    q2 = solve_ik(robot, pregrasp, robot.q, "Pre-grasp")
    animate_joint_move(robot, env, recorder, q2, "Pre-grasp")
    approach_handle(robot, env, recorder, contact)
    open_door(robot, door, env, recorder)
    print("Door fully opened")
    analyse_kinematics(robot, recorder, robot_y)
    while True:
        env.step(0.05)


if __name__ == "__main__":
    main()
