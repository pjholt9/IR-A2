"""Teach door-handle disengagement and return path, without modifying the working open_door().

Start with STAGE = 1. Observe Swift after each stage before increasing STAGE.
This is visual validation, NOT collision detection.
"""
from math import pi
import numpy as np
import swift
from spatialmath import SE3
from scipy import linalg
from scipy.spatial.transform import Rotation
from roboticstoolbox import jtraj

from scenery import Scenery
from robot_tasks import RobotTasks
from kinematic_analysis import analyse_full_kinematics

PRINTER_NUMBER = 1  # 1 or 2
STAGE = 3           # 1 = retract RMRC; 2 = clearance waypoint; 3 = standby

RETRACT_DISTANCE = 0.10  # metres along -EE local Z at the OPEN handle
# TEACH THIS OFFSET AFTER INSPECTING STAGE 1. World XYZ, not printer local.
CLEARANCE_OFFSET_WORLD = np.array([-0.20, 0.20, 0.20])

RMRC_STEPS = 150
RMRC_DT = 0.03
DAMPING = 0.001
KP = 2.0
KR = 2.0
JTRAJ_STEPS = 200
JTRAJ_DT = 0.02


def move_jtraj(robot, env, recorder, q_goal, label):
    qlim = np.asarray([link.qlim for link in robot.links], dtype=float)
    trajectory = jtraj(robot.q.copy(), q_goal, JTRAJ_STEPS).q
    for q in trajectory[1:]:
        if np.any(q < qlim[:, 0]) or np.any(q > qlim[:, 1]):
            raise RuntimeError(f"Joint limit violation during {label}")
        robot.q = q.copy()
        env.step(JTRAJ_DT)
        recorder.record(JTRAJ_DT, label)


def retract_from_open_handle(robot, env, recorder, door):
    """6D RMRC retreat along the same local -Z axis used for pre-grasp.

    Keep the door fixed at -90 deg. No assumptions about world XYZ direction.
    """
    T_handle_open = door.get_handle_pose() @ SE3.Rx(-pi / 2)
    T_goal = T_handle_open @ SE3(0, 0, -RETRACT_DISTANCE)
    T_start = robot.fkine(robot.q)
    pos_error = np.linalg.norm(T_start.t - T_handle_open.t)
    print(f"Open-handle starting position error: {pos_error*1000:.2f} mm")
    if pos_error > 0.005:
        raise RuntimeError("EE is not at the open handle; aborting retract")

    positions = np.linspace(T_start.t, T_goal.t, RMRC_STEPS)
    qlim = np.asarray([link.qlim for link in robot.links], dtype=float)
    for i in range(RMRC_STEPS - 1):
        actual = robot.fkine(robot.q)
        x_des = positions[i + 1]
        v = (positions[i + 1] - positions[i]) / RMRC_DT + KP * (x_des - actual.t)
        w = KR * Rotation.from_matrix(T_goal.R @ actual.R.T).as_rotvec()
        twist = np.r_[v, w]
        J = robot.jacob0(robot.q)
        qdot = J.T @ linalg.solve(J @ J.T + DAMPING * np.eye(6), twist)
        q_next = robot.q + RMRC_DT * qdot
        if np.any(q_next < qlim[:, 0]) or np.any(q_next > qlim[:, 1]):
            raise RuntimeError(f"Retract RMRC joint limit at step {i}")
        robot.q = q_next
        env.step(RMRC_DT)
        recorder.record(RMRC_DT, "Retract")

    T_final = robot.fkine(robot.q)
    pos = np.linalg.norm(T_goal.t - T_final.t)
    rot = np.linalg.norm(Rotation.from_matrix(T_goal.R @ T_final.R.T).as_rotvec())
    print(f"Retract target error: {pos*1000:.2f} mm, {np.rad2deg(rot):.3f} deg")
    if pos > 0.005 or rot > np.deg2rad(2):
        raise RuntimeError("Retract tracking tolerance exceeded")
    return T_goal


def main():
    env = swift.Swift()
    env.launch(realtime=True)
    scenery = Scenery()
    scenery.add_to_env(env)
    printer = scenery.printer1 if PRINTER_NUMBER == 1 else scenery.printer2
    tasks = RobotTasks(scenery, env)
    recorder = tasks.open_door(printer)
    robot = scenery.robot
    door = printer.door
    print("Door opened; door angle should remain -90 deg")

    T_retract = retract_from_open_handle(robot, env, recorder, door)
    print("STAGE 1 DONE: inspect EE/door clearance in Swift.")

    if STAGE >= 2:
        # World XYZ offset is only a TEACHING CANDIDATE; inspect path before STAGE 3.
        T_clear = SE3.Rt(T_retract.R, T_retract.t + CLEARANCE_OFFSET_WORLD)
        sol = robot.ikine_LM(T_clear, q0=robot.q.copy(), joint_limits=True)
        if not sol.success:
            raise RuntimeError("Clearance waypoint IK failed")
        move_jtraj(robot, env, recorder, sol.q, "Clearance waypoint")
        print("STAGE 2 DONE: inspect entire swept path, not just final EE pose.")

    if STAGE >= 3:
        move_jtraj(robot, env, recorder, scenery.ec66.q_standby.copy(), "Return standby")
        print("STAGE 3 DONE: robot at standby, door remains open.")

    print("Door remains open for future camera / retrieval tasks.")
    animation = analyse_full_kinematics(robot, recorder)
    while True:
        env.step(0.05)


if __name__ == "__main__":
    main()
