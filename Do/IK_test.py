
import swift
import numpy as np
from spatialmath import SE3
from math import pi
from scipy import linalg
from scenery import Scenery
from roboticstoolbox import jtraj


env = swift.Swift()
env.launch(realtime=True)

scenery = Scenery()
scenery.add_to_env(env)

env.step(1)

# -------------------------------------------------- #
# Move platform to Station
# -------------------------------------------------- #

scenery.move2(env)

robot = scenery.robot
door = scenery.printer1.door

# Start from standby
robot.q = scenery.ec66.q_standby.copy()
door.update_angle(0)

env.step(1)


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

print("Reached pre-grasp")
env.step(0.5)


# -------------------------------------------------- #
# Phase 2: Pre-grasp -> Handle (Closed-loop RMRC)
# -------------------------------------------------- #

steps = 150
delta_t = 0.03
damping = 0.001
Kp = 2.0

start_pos = robot.fkine(robot.q).t.copy()
end_pos = T_contact.t.copy()

positions = np.linspace(start_pos, end_pos, steps)

for i in range(steps - 1):

    x_desired = positions[i + 1]
    xdot_desired = (positions[i + 1] - positions[i]) / delta_t

    x_actual = robot.fkine(robot.q).t
    error = x_desired - x_actual

    xdot = xdot_desired + Kp * error

    J = robot.jacob0(robot.q)[:3, :]

    qdot = J.T @ linalg.solve(
        J @ J.T + damping * np.eye(3),
        xdot
    )

    q_next = robot.q + delta_t * qdot

    within_limits = all(
        robot.links[j].qlim[0] <= q_next[j] <= robot.links[j].qlim[1]
        for j in range(robot.n)
    )

    if not within_limits:
        raise RuntimeError("RMRC stopped: joint limit reached")

    robot.q = q_next
    env.step(delta_t)

position_error = np.linalg.norm(
    T_contact.t - robot.fkine(robot.q).t
)

print("Handle position error:", position_error)

# Stop if EE is too far from handle
if position_error > 0.005:
    raise RuntimeError("EE did not reach handle accurately")

print("Reached handle")
env.step(0.5)


# -------------------------------------------------- #
# Phase 3: Open door (Continuous IK trajectory)
# -------------------------------------------------- #

angles = np.linspace(0, -pi/2, 91)

q_solutions = []
q_guess = robot.q.copy()

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

print("Door fully opened!")
env.step(1)


# -------------------------------------------------- #
# Keep Swift running
# -------------------------------------------------- #

while True:
    env.step(0.05)
