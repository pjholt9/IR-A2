
import swift
import numpy as np
from spatialmath import SE3
from math import pi
from scenery import Scenery


env = swift.Swift()
env.launch(realtime=True)

scenery = Scenery()
scenery.add_to_env(env)

env.step(1)

# Move platform to current station
scenery.move2(env)

robot = scenery.robot
door = scenery.printer1.door

# Start from the current standby configuration
robot.q = scenery.ec66.q_standby.copy()

# Door opening angles

# -------------------------------------------------- #
# Generate IK trajectory for door opening
# -------------------------------------------------- #

angles = np.linspace(0, -pi/2, 91)

q_solutions = []
q_guess = robot.q.copy()

for angle in angles:

    door.update_angle(angle)

    T_handle = door.get_handle_pose()
    T_target = T_handle @ SE3.Rx(-pi/2)

    solution = robot.ikine_LM(
        T_target,
        q0=q_guess,
        joint_limits=True
    )

    if solution.success:
        q_solutions.append(solution.q.copy())
        q_guess = solution.q.copy()
    else:
        print("IK failed at:", np.rad2deg(angle))
        break

# Reset door before animation
door.update_angle(0)

print(f"Successful poses: {len(q_solutions)}/{len(angles)}")

# -------------------------------------------------- #
# Animate door and robot together
# -------------------------------------------------- #

if len(q_solutions) == len(angles):

    # Place robot at initial handle pose for testing
    robot.q = q_solutions[0].copy()
    env.step(1)

    # Open door
    for i in range(len(angles)):
        door.update_angle(angles[i])
        robot.q = q_solutions[i].copy()
        env.step(0.05)

    env.step(1)

    # Close door (reverse trajectory)
    for i in range(len(angles)-1, -1, -1):
        door.update_angle(angles[i])
        robot.q = q_solutions[i].copy()
        env.step(0.05)

    print("Door animation completed!")

while True:
    env.step(0.05)

