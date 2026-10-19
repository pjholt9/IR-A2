"""Interactive 6-DoF EE/camera pose teaching relative to printer.plate_reference.

Place in Do/ next to scenery.py and robot_tasks.py. The marker represents the
EE frame (camera-to-EE transform is assumed identity during teaching).

Sliders only update the target marker and RGB axes. TEST IK checks a candidate;
TEST & MOVE checks IK then executes a joint-space path. Neither is a collision
checker: visually inspect every robot link, printer, and door in Swift.
"""
import numpy as np
import swift
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from spatialmath import SE3
from spatialgeometry import Sphere, Cylinder
from scipy import linalg
from scipy.spatial.transform import Rotation
from roboticstoolbox import jtraj
from kinematic_analysis import KinematicRecorder

from scenery import Scenery
from robot_tasks import RobotTasks

PRINTER_NUMBER = 1  # 1 or 2
INITIAL_OFFSET_M = [-0.125, -0.150, 0.225]  # relative to plate_reference
INITIAL_RPY_DEG = [-120.0, 0.0, 0.0]
INITIAL_WAYPOINT_WORLD_OFFSET = [-0.15, 0, 0.15]
AXIS_LENGTH_M = 0.075
AXIS_RADIUS_M = 0.003
TRAJECTORY_STEPS = 200
STEP_DT = 0.02
DWELL_SECONDS = 2.0


def as_matrix(T):
    """Swift geometry requires one numeric 4x4 ndarray, not a list/SE3 batch."""
    a = np.asarray(T.A if isinstance(T, SE3) else T, dtype=float)
    if a.shape != (4, 4):
        raise ValueError(f"Expected 4x4 pose; got {a.shape}")
    return a


def make_target(printer, xyz, rpy_deg):
    # Translation and rotation are expressed in the plate-reference frame.
    r, p, y = np.deg2rad(rpy_deg)
    return (printer.plate_reference @ SE3(*xyz)
            @ SE3.Rx(r) @ SE3.Ry(p) @ SE3.Rz(y))


class FrameMarker:
    """Sphere + RGB cylinders; each cylinder's local Z is its axis."""
    def __init__(self, env, pose):
        self.center = Sphere(radius=0.012, pose=SE3(pose.A), color=[1, 1, 1, 1])
        self.center._collision = False
        env.add(self.center)
        self.axes = []
        colors = ([1, 0.1, 0.1, 1], [0.1, 1, 0.1, 1], [0.1, 0.4, 1, 1])
        rotations = (SE3.Ry(np.pi / 2), SE3.Rx(-np.pi / 2), SE3())
        for color, rot in zip(colors, rotations):
            axis = Cylinder(radius=AXIS_RADIUS_M, length=AXIS_LENGTH_M,
                            pose=SE3(), color=color)
            axis._collision = False
            env.add(axis)
            self.axes.append((axis, rot))
        self.update(pose)

    def update(self, pose):
        self.center.T = as_matrix(pose)
        for axis, rot in self.axes:
            # Cylinders are centered; put their near end at the marker origin.
            T = pose @ rot @ SE3(0, 0, AXIS_LENGTH_M / 2)
            axis.T = as_matrix(T)


def analyse_camera_kinematics(robot, recorder):
    """Evaluate both camera motion phases, not only the end configurations.

    Uses the same 6D Jacobian manipulability and sigma_min definitions as
    kinematic_analysis.py. This does NOT validate collision clearance.
    """
    t = np.asarray(recorder.time, dtype=float)
    q = np.asarray(recorder.q, dtype=float)
    phases = np.asarray(recorder.phase)
    J = [robot.jacob0(qi) for qi in q]
    sigma = np.array([np.linalg.svd(Ji, compute_uv=False)[-1] for Ji in J])
    manip = np.array([robot.manipulability(qi) for qi in q])
    qlim = np.array([link.qlim for link in robot.links], dtype=float)
    qdot = np.rad2deg(np.diff(q, axis=0) / np.diff(t)[:, None])
    tdot = (t[1:] + t[:-1]) / 2
    labels = ("Camera waypoint", "Camera RMRC")
    print("\n========== CAMERA KINEMATIC ANALYSIS ==========")
    for label in labels:
        idx = np.flatnonzero(phases == label)
        if not len(idx):
            continue
        worst = idx[np.argmin(sigma[idx])]
        v_idx = idx - 1  # each recorded q has a preceding velocity interval
        print(f"\n{label}:")
        print(f"  Duration: {t[idx[-1]] - t[idx[0]] + (t[idx[0]] - t[idx[0]-1]):.2f} s")
        print(f"  Minimum 6D manipulability: {np.min(manip[idx]):.6f}")
        print(f"  Median 6D manipulability:  {np.median(manip[idx]):.6f}")
        print(f"  Minimum 6D singular value: {sigma[worst]:.6f} at {t[worst]:.2f} s")
        print(f"  Maximum joint speed: {np.max(np.abs(qdot[v_idx])):.2f} deg/s")
        print(f"  Worst q (deg): {np.round(np.rad2deg(q[worst]), 2)}")
    print(f"\nJoint limit violations: {bool(np.any((q < qlim[:,0]) | (q > qlim[:,1])))}")
    print("================================================\n")

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle("EC66 Camera Approach - Two-Stage Kinematic Verification")
    ax[0].plot(t, manip)
    ax[0].set_ylabel("6D manipulability")
    ax[1].plot(t, sigma)
    ax[1].set_ylabel("Minimum singular value (6D)")
    for j in range(robot.n):
        ax[2].plot(tdot, qdot[:, j], label=f"J{j+1}")
    ax[2].set_ylabel("Joint speed (deg/s)")
    ax[2].set_xlabel("Time (s)")
    ax[2].legend(ncol=3, fontsize=8)
    boundary = np.flatnonzero(phases == "Camera RMRC")
    if len(boundary):
        for a in ax:
            a.axvline(t[boundary[0]], color="gray", ls="--", lw=1)
    for a in ax:
        a.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.canvas.draw_idle()
    plt.show(block=False)


def main():
    env = swift.Swift()
    env.launch(realtime=True)
    scenery = Scenery()
    scenery.add_to_env(env)
    env.step(0.1)
    printer = scenery.printers[PRINTER_NUMBER - 1]
    tasks = RobotTasks(scenery, env)

    # Run validated door-opening and return sequence, leaving the door open.
    tasks.open_door(printer)
    tasks.retract_from_door(printer)
    tasks.move_to_clearance(printer)
    tasks.reorient_for_standby(printer)
    tasks.return_to_standby(printer)
    print('Door open; robot at standby. Camera teaching ready.')

    robot = scenery.robot
    ref = printer.plate_reference
    # Initial slider angles match the current EE orientation in plate frame.
    R_relative = ref.R.T @ robot.fkine(robot.q).R
    init_rpy = np.array(INITIAL_RPY_DEG, dtype=float)
    initial_pose = make_target(printer, INITIAL_OFFSET_M, init_rpy)
    marker = FrameMarker(env, initial_pose)
    waypoint_marker = FrameMarker(
        env, SE3.Rt(initial_pose.R,
                    initial_pose.t + np.array(INITIAL_WAYPOINT_WORLD_OFFSET)))

    fig = plt.figure(figsize=(11.0, 9.2))
    fig.suptitle('Camera / EE pose teaching | Plate-reference frame | RGB = XYZ')
    info = fig.add_axes([0.10, 0.81, 0.85, 0.13])
    info.axis('off')
    status = info.text(0, 0.95, 'Move sliders; marker only. TEST IK before TEST & MOVE.',
                       fontsize=10, va='top')
    coordinates = info.text(0, 0.52, '', fontsize=9, family='monospace', va='top')

    sliders = []
    specs = [
        ('X local (m)', -0.40, 0.40, INITIAL_OFFSET_M[0], '%.3f'),
        ('Y local (m)', -0.40, 0.40, INITIAL_OFFSET_M[1], '%.3f'),
        ('Z local (m)', -0.10, 0.60, INITIAL_OFFSET_M[2], '%.3f'),
        ('Roll X (deg)', -180, 180, float(init_rpy[0]), '%.1f'),
        ('Pitch Y (deg)', -180, 180, float(init_rpy[1]), '%.1f'),
        ('Yaw Z (deg)', -180, 180, float(init_rpy[2]), '%.1f'),
    ]
    for i, (name, lo, hi, start, fmt) in enumerate(specs):
        ax = fig.add_axes([0.20, 0.75 - 0.067 * i, 0.64, 0.035])
        sliders.append(Slider(ax, name, lo, hi, valinit=start, valfmt=fmt))

    # Waypoint offset is in WORLD XYZ from the camera target.
    # Its orientation is the same as the target, so RMRC can approach
    # the camera along a straight Cartesian line without wrist rotation.
    waypoint_sliders = []
    for i, (axis_name, initial) in enumerate(
            zip(("WX world offset", "WY world offset", "WZ world offset"),
                INITIAL_WAYPOINT_WORLD_OFFSET)):
        ax = fig.add_axes([0.20, 0.29 - 0.055 * i, 0.64, 0.025])
        waypoint_sliders.append(
            Slider(ax, axis_name + " (m)", -0.40, 0.40,
                   valinit=initial, valfmt="%.3f"))

    def candidate():
        values = [float(s.val) for s in sliders]
        return make_target(printer, values[:3], values[3:]), values

    def waypoint_target(T_camera):
        offset = np.array([float(s.val) for s in waypoint_sliders])
        return SE3.Rt(T_camera.R, T_camera.t + offset), offset

    def update(_=None):
        T, vals = candidate()
        T_way, offset = waypoint_target(T)
        marker.update(T)
        waypoint_marker.update(T_way)
        coordinates.set_text(
            f'Plate offset XYZ: {np.round(vals[:3], 4)} m\n'
            f'Local RPY (Rx Ry Rz): {np.round(vals[3:], 1)} deg  |  '
            f'World XYZ: {np.round(T.t, 4)} m\n'
            f'Waypoint world offset: {np.round(offset, 3)} m')
        fig.canvas.draw_idle()
        env.step(0.001)

    for slider in sliders + waypoint_sliders:
        slider.on_changed(update)

    def solve():
        T_cam, vals = candidate()
        T_way, offset = waypoint_target(T_cam)
        q_start = robot.q.copy()
        sol_way = robot.ikine_LM(T_way, q0=q_start, joint_limits=True)
        if not sol_way.success:
            status.set_text('WAYPOINT IK FAILED. Adjust waypoint sliders.')
            fig.canvas.draw_idle()
            return None
        sol_cam = robot.ikine_LM(T_cam, q0=sol_way.q, joint_limits=True)
        if not sol_cam.success:
            status.set_text('CAMERA IK FAILED from waypoint. Adjust sliders.')
            fig.canvas.draw_idle()
            return None
        for name, sol, T in (('Waypoint', sol_way, T_way),
                             ('Camera', sol_cam, T_cam)):
            actual = robot.fkine(sol.q)
            p_err = np.linalg.norm(actual.t - T.t)
            a_err = np.linalg.norm(
                Rotation.from_matrix(T.R @ actual.R.T).as_rotvec())
            if p_err > 0.005 or a_err > np.deg2rad(2):
                status.set_text(f'{name} IK residual too large.')
                fig.canvas.draw_idle()
                return None
        print('Waypoint IK and camera IK: SUCCESS')
        print('Camera XYZ/RPY (plate frame):', np.round(vals, 4))
        print('Waypoint offset (world):', np.round(offset, 4))
        status.set_text('Both IK targets valid. Path NOT collision checked.')
        fig.canvas.draw_idle()
        return T_cam, T_way, sol_way.q.copy(), vals, offset

    def plan_rmrc(q_initial, T_target, steps=180, dt=0.03):
        """Precompute a 6D DLS RMRC path WITHOUT moving the robot."""
        q = q_initial.copy()
        T_initial = robot.fkine(q)
        qlim = np.array([link.qlim for link in robot.links], dtype=float)
        path = []
        positions = np.linspace(T_initial.t, T_target.t, steps)
        for i in range(steps - 1):
            actual = robot.fkine(q)
            x_next = positions[i + 1]
            v = (x_next - positions[i]) / dt + 2.0 * (x_next - actual.t)
            w = 2.0 * Rotation.from_matrix(
                T_target.R @ actual.R.T).as_rotvec()
            J = robot.jacob0(q)
            qdot = J.T @ linalg.solve(
                J @ J.T + 0.001 * np.eye(6), np.r_[v, w])
            q_next = q + dt * qdot
            if not np.all(np.isfinite(q_next)):
                raise RuntimeError('RMRC produced invalid joint values')
            if np.any(q_next < qlim[:, 0]) or np.any(q_next > qlim[:, 1]):
                raise RuntimeError(f'RMRC joint limit at step {i}')
            if np.rad2deg(np.max(np.abs(q_next - q))) > 5.0:
                raise RuntimeError(f'RMRC joint jump at step {i}')
            path.append(q_next.copy())
            q = q_next
        actual = robot.fkine(q)
        p_err = np.linalg.norm(actual.t - T_target.t) * 1000
        a_err = np.rad2deg(np.linalg.norm(
            Rotation.from_matrix(T_target.R @ actual.R.T).as_rotvec()))
        if p_err > 5.0 or a_err > 2.0:
            raise RuntimeError(
                f'RMRC terminal error {p_err:.2f} mm, {a_err:.2f} deg')
        return path, p_err, a_err

    def test_ik(_):
        solve()

    def test_move(_):
        result = solve()
        if result is None:
            return
        T_cam, T_way, q_way, vals, offset = result
        q_start = robot.q.copy()
        path1 = jtraj(q_start, q_way, TRAJECTORY_STEPS).q
        limits = np.array([link.qlim for link in robot.links], dtype=float)
        if np.any(path1 < limits[:, 0]) or np.any(path1 > limits[:, 1]):
            status.set_text('ABORT: joint limit along waypoint jtraj.')
            fig.canvas.draw_idle()
            return
        try:
            path2, p_err, a_err = plan_rmrc(q_way, T_cam)
        except Exception as exc:
            print('Camera RMRC preflight failed:', exc)
            status.set_text(f'ABORT: {exc}')
            fig.canvas.draw_idle()
            return
        print('Preflight passed. Executing waypoint jtraj then camera RMRC.')
        recorder = KinematicRecorder(robot)
        # IMPORTANT: This does not guarantee collision-free motion.
        for q in path1[1:]:
            robot.q = q.copy()
            env.step(STEP_DT)
            recorder.record(STEP_DT, "Camera waypoint")
        for q in path2:
            robot.q = q.copy()
            env.step(0.03)
            recorder.record(0.03, "Camera RMRC")
        analyse_camera_kinematics(robot, recorder)
        print(f'CAMERA: position error {p_err:.2f} mm; orientation error {a_err:.3f} deg')
        print('SAVED PLATE-RELATIVE CAMERA TARGET:')
        print(f'  XYZ (m): {np.round(vals[:3], 5).tolist()}')
        print(f'  RPY (deg): {np.round(vals[3:], 4).tolist()}')
        print(f'  Waypoint WORLD offset from camera: {np.round(offset, 5).tolist()}')
        print('  T_plate_reference_to_EE =\\n', ref.inv() @ T_cam)
        status.set_text('AT CAMERA POSE. Inspect clearance. Camera capture dwell.')
        fig.canvas.draw_idle()
        env.step(DWELL_SECONDS)

    ax_ik = fig.add_axes([0.18, 0.045, 0.25, 0.045])
    ax_move = fig.add_axes([0.56, 0.045, 0.28, 0.045])
    btn_ik = Button(ax_ik, 'TEST IK')
    btn_move = Button(ax_move, 'TEST & MOVE')
    btn_ik.on_clicked(test_ik)
    btn_move.on_clicked(test_move)
    fig._camera_teach_buttons = (btn_ik, btn_move)
    update()
    plt.show()
    print('Teaching finished. Door remains open.')
    while True:
        env.step(0.05)


if __name__ == '__main__':
    main()
