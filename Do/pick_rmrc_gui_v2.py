"""IR-A2: GUI for straight-line RMRC pick and reverse retreat (Swift only).

Put next to scenery.py and robot_tasks.py in Do/. Does not edit core files.
Run from IR-A2: .\\.venv\\Scripts\\python.exe Do\\pick_rmrc_gui.py

No mesh collision checker. No physical gripper/part attachment.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import swift
from spatialmath import SE3
from scipy import linalg
from scipy.spatial.transform import Rotation
from roboticstoolbox import jtraj
from scenery import Scenery
from robot_tasks import RobotTasks

DT = 0.03
STEPS = 220
DAMPING = 0.002
KP = 2.0
KR = 2.0
POS_TOL = 0.005
ROT_TOL = np.deg2rad(2.0)


def rotation_error(R_target, R_actual):
    return Rotation.from_matrix(R_target @ R_actual.T).as_rotvec()


def offline_rmrc(robot, q_start, T_target, steps=STEPS, dt=DT):
    """Plan in joint space WITHOUT commanding Swift; reject limits/tracking errors.

    Desired Cartesian XYZ follows a line. Orientation stays fixed at start.
    """
    q = np.asarray(q_start, dtype=float).copy()
    T_start = robot.fkine(q)
    R_fixed = T_start.R.copy()
    if np.linalg.norm(rotation_error(T_target.R, R_fixed)) > ROT_TOL:
        raise RuntimeError('Linear segment requires the same orientation at both ends')
    xyz = np.linspace(T_start.t, T_target.t, steps)
    qlim = np.asarray([link.qlim for link in robot.links], dtype=float)
    path = [q.copy()]
    for i in range(steps - 1):
        T_now = robot.fkine(q)
        v = (xyz[i + 1] - xyz[i]) / dt + KP * (xyz[i + 1] - T_now.t)
        w = KR * rotation_error(R_fixed, T_now.R)
        J = robot.jacob0(q)
        qdot = J.T @ linalg.solve(J @ J.T + DAMPING * np.eye(6), np.r_[v, w])
        q_next = q + dt * qdot
        if not np.all(np.isfinite(q_next)):
            raise RuntimeError(f'RMRC non-finite solution at step {i}')
        if np.any(q_next < qlim[:, 0]) or np.any(q_next > qlim[:, 1]):
            raise RuntimeError(f'RMRC joint limit at step {i}')
        if np.max(np.abs(q_next - q)) > np.deg2rad(7):
            raise RuntimeError(f'RMRC joint jump >7 deg at step {i}')
        q = q_next
        path.append(q.copy())
    T_end = robot.fkine(q)
    pos_err = np.linalg.norm(T_target.t - T_end.t)
    rot_err = np.linalg.norm(rotation_error(T_target.R, T_end.R))
    if pos_err > POS_TOL or rot_err > ROT_TOL:
        raise RuntimeError(f'RMRC tracking failed: {pos_err*1000:.2f} mm / {np.rad2deg(rot_err):.2f} deg')
    return np.asarray(path), pos_err, rot_err


def main():
    env = swift.Swift()
    env.launch(realtime=True)
    scenery = Scenery()
    scenery.add_to_env(env)
    env.step(0.1)
    robot = scenery.robot
    tasks = RobotTasks(scenery, env)
    part = scenery.part1
    marker = scenery.part1_pick_marker
    printer = scenery.printer1
    sizes = {'barrel': (0.050, 0.240), 'body': (0.07560, 0.17498)}
    width, height = sizes[part.kind]
    state = {'prepared': False, 'plan': None, 'finished': False}

    root = tk.Tk()
    root.title(f'IR-A2 | {part.kind.upper()} | RMRC pick + retreat')
    root.geometry('750x830')
    ttk.Label(root, text=f'{part.kind.upper()} | RMRC linear pick and retreat',
              font=('Segoe UI', 14, 'bold')).pack(pady=(12, 5))
    ttk.Label(root, text='STL-local pick XYZ (mm) | EE orientation relative to printer.plate_reference (deg)').pack()
    ttk.Label(root, text='Simulation only: NO collision checking; NO part attachment', foreground='#a14b00').pack(pady=5)
    status = tk.StringVar(value='1. Prepare robot, 2. Preview, 3. Execute.')
    vars_ = {}
    readouts = {}

    def invalidate(*_):
        state['plan'] = None
        if not state['finished']:
            status.set('Pose changed. Preview again before executing.')

    def slider(frame, name, lo, hi, default, unit, on_change):
        line = ttk.Frame(frame)
        line.pack(fill='x', padx=12, pady=5)
        ttk.Label(line, text=name, width=8).pack(side='left')
        var = tk.DoubleVar(value=default)
        vars_[name] = var
        ttk.Scale(line, from_=lo, to=hi, variable=var, command=on_change).pack(side='left', fill='x', expand=True, padx=10)
        label = tk.StringVar(value=f'{default:.1f}')
        readouts[name] = label
        ttk.Label(line, textvariable=label, width=9).pack(side='left')
        ttk.Label(line, text=unit, width=5).pack(side='left')

    def update_position(*_):
        if not all(k in vars_ for k in ('X', 'Y', 'Z')):
            return
        xyz = [vars_[k].get() / 1000 for k in ('X', 'Y', 'Z')]
        marker.T = (part.pose @ SE3(*xyz)).A
        for k in ('X', 'Y', 'Z'):
            readouts[k].set(f'{vars_[k].get():.1f}')
        invalidate()
        env.step(0.01)

    def update_rotation(*_):
        for k in ('Rx', 'Ry', 'Rz'):
            if k in vars_:
                readouts[k].set(f'{vars_[k].get():.1f}')
        invalidate()

    pos_frame = ttk.LabelFrame(root, text='Pick marker (STL local)')
    pos_frame.pack(fill='x', padx=14, pady=5)
    slider(pos_frame, 'X', -30, 120, width * 500, 'mm', update_position)
    slider(pos_frame, 'Y', -80, 120, 0, 'mm', update_position)
    slider(pos_frame, 'Z', 0, 270, height * 2000 / 3, 'mm', update_position)
    rot_frame = ttk.LabelFrame(root, text='EE orientation (plate frame)')
    rot_frame.pack(fill='x', padx=14, pady=5)
    slider(rot_frame, 'Rx', -180, 180, -90, 'deg', update_rotation)
    slider(rot_frame, 'Ry', -180, 180, 0, 'deg', update_rotation)
    slider(rot_frame, 'Rz', -180, 180, 0, 'deg', update_rotation)
    distance_frame = ttk.LabelFrame(root, text='Straight approach / retreat')
    distance_frame.pack(fill='x', padx=14, pady=5)
    slider(distance_frame, 'Approach offset', 150, 350, 250, 'mm', invalidate)
    slider(distance_frame, 'Retreat distance', 300, 650, 600, 'mm', invalidate)

    def pick_pose():
        p = np.asarray(marker.T, dtype=float)[:3, 3]
        rx, ry, rz = (np.deg2rad(vars_[k].get()) for k in ('Rx', 'Ry', 'Rz'))
        R = printer.plate_reference.R @ SE3.Rx(rx).R @ SE3.Ry(ry).R @ SE3.Rz(rz).R
        return SE3.Rt(R, p)

    def prepare():
        if state['prepared'] or state['finished']:
            messagebox.showinfo('Already started', 'Restart this script to run a new test.')
            return
        if not messagebox.askyesno('Prepare robot', 'Run door → retract → clearance → standby → camera → pre-pick?'):
            return
        try:
            tasks.open_door(printer)
            tasks.retract_from_door(printer)
            tasks.move_to_clearance(printer)
            tasks.reorient_for_standby(printer)
            tasks.return_to_standby(printer)
            tasks.move_to_camera_pose(printer)
            tasks.move_to_pre_pick(printer, distance=0.10)
            state['prepared'] = True
            status.set('Prepared. Preview linear path and IK.')
        except Exception as exc:
            messagebox.showerror('Prepare failed', str(exc))

    def preview():
        if not state['prepared'] or state['finished']:
            messagebox.showwarning('Not ready', 'Prepare robot first (or restart after completing a run).')
            return
        try:
            T_pick = pick_pose()
            approach_offset = vars_['Approach offset'].get() / 1000.0
            retreat_distance = vars_['Retreat distance'].get() / 1000.0
            if retreat_distance < approach_offset:
                raise ValueError('Retreat distance must be >= approach offset')
            # Both waypoints lie on the same WORLD Y line through the pick target.
            T_out = SE3.Rt(T_pick.R, T_pick.t + np.array([0.0, approach_offset, 0.0]))
            T_retreat = SE3.Rt(T_pick.R, T_pick.t + np.array([0.0, retreat_distance, 0.0]))
            sol = robot.ikine_LM(T_out, q0=robot.q.copy(), joint_limits=True)
            if not sol.success:
                raise RuntimeError('Outside alignment pose IK failed')
            q_out = np.asarray(sol.q, dtype=float)
            # Joint interpolation only to outside alignment, then Cartesian RMRC.
            align_path = np.asarray(jtraj(robot.q.copy(), q_out, 180).q)
            qlim = np.asarray([link.qlim for link in robot.links], dtype=float)
            if np.any(align_path < qlim[:, 0]) or np.any(align_path > qlim[:, 1]):
                raise RuntimeError('Outside alignment violates joint limits')
            T_aligned = robot.fkine(q_out)
            if np.linalg.norm(T_aligned.t - T_out.t) > POS_TOL:
                raise RuntimeError('Outside alignment IK XYZ error too high')
            inward, pos_in, rot_in = offline_rmrc(robot, q_out, T_pick)
            # Retreat follows the same Cartesian line in reverse, extending past alignment.
            outward, pos_out, rot_out = offline_rmrc(robot, inward[-1], T_retreat, steps=340)
            q_standby = np.asarray(scenery.ec66.q_standby, dtype=float)
            standby_path = np.asarray(jtraj(outward[-1], q_standby, 200).q)
            if np.any(standby_path < qlim[:, 0]) or np.any(standby_path > qlim[:, 1]):
                raise RuntimeError('Standby return violates joint limits')
            state['plan'] = (align_path, inward, outward, standby_path)
            status.set(f'PREVIEW OK | approach {approach_offset*1000:.0f} mm | retreat {retreat_distance*1000:.0f} mm | errors {pos_in*1000:.1f}/{pos_out*1000:.1f} mm')
            print('Pick WORLD XYZ (m):', np.round(T_pick.t, 5))
            print('Outside WORLD XYZ (m):', np.round(T_out.t, 5))
            print('Approach offset (m):', approach_offset)
            print('Retreat distance (m):', retreat_distance)
            print('Retreat WORLD XYZ (m):', np.round(T_retreat.t, 5))
            print('RMRC tracking errors (mm):', round(pos_in*1000, 2), round(pos_out*1000, 2))
            print('WARNING: IK and joint-limit checks do NOT establish collision clearance.')
        except Exception as exc:
            state['plan'] = None
            messagebox.showerror('Preview failed', str(exc))

    def execute():
        if state['plan'] is None or state['finished']:
            messagebox.showwarning('No valid plan', 'Prepare and preview before executing.')
            return
        if not messagebox.askyesno('SIMULATION ONLY — no collision checking',
                'Execute: joint-space align OUTSIDE → RMRC straight IN → hold at pick → '
                'RMRC straight OUT beyond alignment → joint-space standby?\n\n'
                'Mesh collision is NOT checked. The part is NOT attached to the gripper.'):
            return
        plan = state['plan']
        state['plan'] = None
        state['finished'] = True
        recorder = getattr(tasks, '_active_recorder', None)
        try:
            for name, path in zip(('Outside alignment', 'RMRC approach', 'RMRC retreat', 'Return standby'), plan):
                print(name)
                for q in path[1:]:
                    robot.q = q.copy()
                    env.step(DT)
                    if recorder is not None:
                        recorder.record(DT, name)
                if name == 'RMRC approach':
                    print('Grasp target reached (virtual grasp; no attachment).')
                    for _ in range(25):
                        env.step(DT)
            status.set('Completed: approach → virtual grasp → reverse retreat → standby. Door remains open.')
        except Exception as exc:
            messagebox.showerror('Motion stopped', str(exc))

    ttk.Button(root, text='1. Prepare: door → camera → pre-pick', command=prepare).pack(fill='x', padx=16, pady=6)
    ttk.Button(root, text='2. Preview full path (IK + RMRC, NO motion)', command=preview).pack(fill='x', padx=16, pady=6)
    ttk.Button(root, text='3. RUN straight approach → retreat → standby', command=execute).pack(fill='x', padx=16, pady=6)
    ttk.Label(root, textvariable=status, wraplength=690, foreground='#075985').pack(padx=16, pady=12)
    ttk.Label(root, text='NOTE: Outside alignment and standby return use jtraj, not RMRC. '
              'Only the straight inward/outward segments use Jacobian DLS RMRC. '
              'No geometry collision checking. Test visually in Swift.',
              wraplength=690).pack(padx=16, pady=4)
    update_position()
    update_rotation()
    root.mainloop()


if __name__ == '__main__':
    main()
