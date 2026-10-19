"""Manual grasp-marker/IK sandbox for the existing IR-A2 Scenery.
Run from IR-A2: .\.venv\Scripts\python.exe Do\pick_waypoint_gui.py
Requires Scenery.part1 and Scenery.part1_pick_marker (from prior marker setup).
No automatic collision checking; keep clear of printer geometry.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import swift
from spatialmath import SE3
from roboticstoolbox import jtraj
from scenery import Scenery
from robot_tasks import RobotTasks


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
    kind = part.kind
    sizes = {'barrel': (0.050, 0.240), 'body': (0.07560, 0.17498)}
    width, height = sizes[kind]
    root = tk.Tk()
    root.title(f'IR-A2 | {kind.upper()} | Pick waypoint sandbox')
    root.geometry('580x540')
    ttk.Label(root, text=f'Part: {kind} | Marker local to STL origin (mm)', font=('Segoe UI', 12, 'bold')).pack(pady=10)
    ttk.Label(root, text='Move marker first. IK preview does not move robot. No automatic collision detection.', foreground='#9b4200').pack()
    sliders = {}
    entries = {}
    limits = [('X', -30, 120, width * 500), ('Y', -80, 120, 0), ('Z', 0, 270, height * 2000 / 3)]
    status = tk.StringVar(value='Ready. Marker follows part.pose @ local offset.')
    q_goal = {'value': None}
    target_pose = {'value': None}

    def update_marker(*_):
        xyz = [sliders[a].get() / 1000.0 for a in ('X', 'Y', 'Z')]
        marker.T = (part.pose @ SE3(*xyz)).A
        for a in ('X', 'Y', 'Z'):
            entries[a].set(f'{sliders[a].get():.1f}')
        q_goal['value'] = None
        target_pose['value'] = None
        status.set('Marker updated. Run IK Preview again before any movement.')
        env.step(0.01)

    for axis, lo, hi, default in limits:
        frame = ttk.Frame(root)
        frame.pack(fill='x', padx=16, pady=8)
        ttk.Label(frame, text=f'{axis} local', width=9).pack(side='left')
        var = tk.DoubleVar(value=default)
        sliders[axis] = var
        scale = ttk.Scale(frame, from_=lo, to=hi, variable=var, command=update_marker)
        scale.pack(side='left', fill='x', expand=True, padx=8)
        entries[axis] = tk.StringVar(value=f'{default:.1f}')
        ttk.Label(frame, textvariable=entries[axis], width=7).pack(side='left')
        ttk.Label(frame, text='mm').pack(side='left')

    def prep_robot():
        if not messagebox.askyesno('Run existing motion', 'Run door opening + retreat + clearance + standby + camera + pre-pick?\nThis moves the robot and is NOT collision checked automatically.'):
            return
        try:
            printer = scenery.printer1
            tasks.open_door(printer)
            tasks.retract_from_door(printer)
            tasks.move_to_clearance(printer)
            tasks.reorient_for_standby(printer)
            tasks.return_to_standby(printer)
            tasks.move_to_camera_pose(printer)
            tasks.move_to_pre_pick(printer, distance=0.10)
            status.set('Pre-pick reached. Preview IK before proceeding.')
        except Exception as exc:
            messagebox.showerror('Prep failed', str(exc))

    def preview_ik():
        try:
            p = np.asarray(marker.T)[:3, 3]
            # Keep the CURRENT EE orientation; position-only marker does not specify gripper orientation.
            T = SE3.Rt(robot.fkine(robot.q).R, p)
            sol = robot.ikine_LM(T, q0=robot.q.copy(), joint_limits=True)
            if not sol.success:
                raise RuntimeError('IK solver did not converge. Adjust marker or orientation.')
            actual = robot.fkine(sol.q)
            error_mm = np.linalg.norm(actual.t - p) * 1000
            q_goal['value'] = np.asarray(sol.q, dtype=float).copy()
            target_pose['value'] = T
            status.set(f'IK SUCCESS | position residual {error_mm:.2f} mm | NO motion yet')
            print('IK target WORLD XYZ (m):', np.round(p, 5))
            print('IK q (deg):', np.round(np.rad2deg(sol.q), 2))
        except Exception as exc:
            q_goal['value'] = None
            messagebox.showerror('IK preview failed', str(exc))

    def move_robot():
        if q_goal['value'] is None:
            messagebox.showwarning('No IK solution', 'Run IK Preview first.')
            return
        if not messagebox.askyesno('UNSAFE motion confirmation',
            'Move robot directly to the marker using joint-space jtraj?\n'
            'WARNING: NO mesh collision checking. Robot may pass through the printer, door, or part.\n'
            'Use ONLY for careful visual simulation inspection.'):
            return
        try:
            traj = jtraj(robot.q.copy(), q_goal['value'], 180).q
            for q in traj[1:]:
                robot.q = np.asarray(q, dtype=float)
                env.step(0.025)
            status.set('Direct joint-space test completed. Inspect visually for intersections.')
        except Exception as exc:
            messagebox.showerror('Motion error', str(exc))

    ttk.Separator(root).pack(fill='x', padx=16, pady=14)
    ttk.Button(root, text='1. Run existing door → pre-pick pipeline', command=prep_robot).pack(fill='x', padx=16, pady=5)
    ttk.Button(root, text='2. IK Preview (no motion)', command=preview_ik).pack(fill='x', padx=16, pady=5)
    ttk.Button(root, text='3. TEST direct jtraj to marker (no collision check)', command=move_robot).pack(fill='x', padx=16, pady=5)
    ttk.Label(root, textvariable=status, wraplength=530, foreground='#075985').pack(padx=16, pady=16)
    ttk.Label(root, text='Use Scenery.part1_type to switch barrel/body. Marker is position only; EE orientation stays current.', wraplength=530).pack(padx=16)
    update_marker()
    root.mainloop()


if __name__ == '__main__':
    main()
