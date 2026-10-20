"""Interactive clearance-waypoint teaching after successful door opening and RMRC retract.

Run this file from Do/. Stage 2 uses Matplotlib XYZ sliders and TEST & MOVE.
Keep STAGE=2 until the entire clearance path is visually verified in Swift.
"""
from math import pi
import numpy as np
import swift
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from spatialgeometry import Sphere
from spatialmath import SE3
from scipy import linalg
from scipy.spatial.transform import Rotation, Slerp
from roboticstoolbox import jtraj

from scenery import Scenery
from Do.robot_tasks_backup import RobotTasks
from kinematic_analysis import analyse_full_kinematics

PRINTER_NUMBER = 1
STAGE = 4  # 1: retract; 2: clearance; 3: reorient; 4: return standby
RETRACT_DISTANCE = 0.10
INITIAL_CLEARANCE_OFFSET_WORLD = np.array([-0.20, 0.30, 0.20])
RMRC_STEPS, RMRC_DT = 150, 0.03
DAMPING, KP, KR = 0.001, 2.0, 2.0
JTRAJ_STEPS, JTRAJ_DT = 200, 0.02
REORIENT_STEPS = 60
MAX_REORIENT_JOINT_STEP_DEG = 8.0


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
        v = ((positions[i+1] - positions[i]) / RMRC_DT
             + KP * (positions[i+1] - actual.t))
        w = KR * Rotation.from_matrix(T_goal.R @ actual.R.T).as_rotvec()
        J = robot.jacob0(robot.q)
        qdot = J.T @ linalg.solve(J @ J.T + DAMPING * np.eye(6), np.r_[v, w])
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


def teach_clearance(robot, env, recorder, T_retract):
    """Teach WORLD XYZ offset relative to the end of retract; orientation held fixed.

    Only TEST & MOVE commands robot motion. Slider changes only move the marker.
    Returns True if a candidate was executed successfully.
    """
    base = np.asarray(T_retract.t, dtype=float)
    marker = Sphere(radius=0.025, pose=SE3(*base))
    marker._collision = False  # Required for this Swift/spatialgeometry version
    env.add(marker)

    fig = plt.figure(figsize=(8.4, 4.0))
    fig.suptitle("Clearance waypoint teaching | WORLD XYZ offset from retract")
    ax_info = fig.add_axes([0.10, 0.62, 0.82, 0.24])
    ax_info.axis("off")
    status = ax_info.text(0.0, 0.85, "Move sliders, inspect marker in Swift, then TEST & MOVE.",
                          fontsize=10, va="top", wrap=True)
    coords = ax_info.text(0.0, 0.32, "", fontsize=10, family="monospace")

    sliders = []
    for label, y, initial in zip(("World X (m)", "World Y (m)", "World Z (m)"),
                                 (0.49, 0.38, 0.27), INITIAL_CLEARANCE_OFFSET_WORLD):
        ax = fig.add_axes([0.22, y, 0.67, 0.055])
        sliders.append(Slider(ax, label, -0.35, 0.35, valinit=float(initial), valstep=0.01))

    ax_button = fig.add_axes([0.35, 0.08, 0.32, 0.11])
    button = Button(ax_button, "TEST & MOVE")
    state = {"moved": False}

    def candidate():
        offset = np.array([s.val for s in sliders], dtype=float)
        return SE3.Rt(T_retract.R, base + offset), offset

    def update(_=None):
        T, offset = candidate()
        marker.T = np.asarray(T.A, dtype=float)
        coords.set_text(f"Offset: {offset.round(3)} m   |   World target: {T.t.round(3)} m")
        fig.canvas.draw_idle()

    def test_and_move(_):
        if state["moved"]:
            status.set_text("Already moved. Close this window to continue.")
            fig.canvas.draw_idle()
            return
        T, offset = candidate()
        status.set_text("Solving IK...")
        fig.canvas.draw_idle()
        try:
            sol = robot.ikine_LM(T, q0=robot.q.copy(), joint_limits=True)
            if not sol.success:
                status.set_text("IK FAILED. Adjust X/Y/Z and test again; robot has NOT moved.")
                fig.canvas.draw_idle()
                print("Clearance waypoint IK failed for offset", offset)
                return
            q_goal = np.asarray(sol.q, dtype=float)
            # A successful endpoint IK does not prove the interpolated motion is collision-free.
            status.set_text("IK success. Executing joint-space path; inspect full arm in Swift.")
            fig.canvas.draw_idle()
            move_jtraj(robot, env, recorder, q_goal, "Clearance waypoint")
            state["moved"] = True
            error = np.linalg.norm(robot.fkine(robot.q).t - T.t) * 1000
            status.set_text(f"MOVED. EE target error: {error:.2f} mm. Inspect links/door; close to continue.")
            print("STAGE 2 DONE. Clearance offset (world m):", offset)
            print(f"Clearance EE position error: {error:.2f} mm")
        except Exception as exc:
            status.set_text(f"Motion/IK error: {exc}")
            print("Stage 2 error:", repr(exc))
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update)
    button.on_clicked(test_and_move)
    update()
    plt.show(block=True)
    return state["moved"]



def reorient_at_clearance(robot, env, recorder, R_target):
    """Rotate EE at its current XYZ, following a Cartesian orientation interpolation.

    IK is preflighted for the full sequence before commanding any motion.
    Does not guarantee link-level collision clearance: inspect in Swift.
    """
    T_start = robot.fkine(robot.q)
    xyz = np.asarray(T_start.t, dtype=float).copy()
    key_rots = Rotation.from_matrix(np.stack((T_start.R, R_target)))
    slerp = Slerp([0.0, 1.0], key_rots)
    q_prev = robot.q.copy()
    q_path = []
    qlim = np.asarray([link.qlim for link in robot.links], dtype=float)

    for fraction in np.linspace(0.0, 1.0, REORIENT_STEPS)[1:]:
        R = slerp([fraction]).as_matrix()[0]
        T_goal = SE3.Rt(R, xyz)
        sol = robot.ikine_LM(T_goal, q0=q_prev, joint_limits=True)
        if not sol.success:
            raise RuntimeError(f"Reorientation IK failed at {fraction:.1%}; no rotation executed")
        q_next = np.asarray(sol.q, dtype=float)
        if np.any(q_next < qlim[:, 0]) or np.any(q_next > qlim[:, 1]):
            raise RuntimeError("Reorientation joint limit violation; no rotation executed")
        step_deg = np.rad2deg(np.max(np.abs(q_next - q_prev)))
        if step_deg > MAX_REORIENT_JOINT_STEP_DEG:
            raise RuntimeError(f"Reorientation IK jump {step_deg:.2f} deg; no rotation executed")
        q_path.append(q_next)
        q_prev = q_next

    for q in q_path:
        robot.q = q.copy()
        env.step(RMRC_DT)
        recorder.record(RMRC_DT, "Reorient EE")

    T_end = robot.fkine(robot.q)
    pos_error = np.linalg.norm(T_end.t - xyz) * 1000
    angle_error = np.rad2deg(np.linalg.norm(
        Rotation.from_matrix(R_target @ T_end.R.T).as_rotvec()))
    print(f"STAGE 3 DONE: reoriented EE at clearance; XYZ drift {pos_error:.2f} mm, "
          f"orientation error {angle_error:.3f} deg")
    print("Inspect wrist/elbow/door clearance in Swift before STAGE 4.")


def main():
    env = swift.Swift()
    env.launch(realtime=True)
    scenery = Scenery()
    scenery.add_to_env(env)
    printer = scenery.printer1 if PRINTER_NUMBER == 1 else scenery.printer2
    # Save the original handle-approach EE orientation while the door is CLOSED.
    R_approach = (printer.door.get_handle_pose() @ SE3.Rx(-pi / 2)).R.copy()
    tasks = RobotTasks(scenery, env)
    recorder = tasks.open_door(printer)
    robot = scenery.robot
    door = printer.door
    print("Door opened; remains at -90 deg")

    T_retract = retract_from_open_handle(robot, env, recorder, door)
    print("STAGE 1 DONE: inspect EE/door clearance in Swift.")

    if STAGE >= 2:
        moved = teach_clearance(robot, env, recorder, T_retract)
        if not moved:
            print("No clearance waypoint executed. Standby skipped.")
        elif STAGE >= 3:
            reorient_at_clearance(robot, env, recorder, R_approach)
            if STAGE >= 4:
                print("WARNING: Returning via joint-space interpolation; visually inspect full path.")
                move_jtraj(robot, env, recorder, scenery.ec66.q_standby.copy(), "Return standby")
                print("STAGE 4 DONE: standby reached; door remains open")

    print("Door remains open for future camera / retrieval tasks.")
    analyse_full_kinematics(robot, recorder)
    while True:
        env.step(0.05)


if __name__ == "__main__":
    main()
