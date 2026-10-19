"""IR-A2 pick waypoint GUI: position marker + 3 orientation sliders.

Run from IR-A2:
    .\.venv\Scripts\python.exe Do\pick_waypoint_gui_orientation.py

Uses existing RobotTasks.move_to_pre_pick() UNCHANGED.
Orientation sliders are relative to printer.plate_reference (degrees),
using R = R_plate @ Rx @ Ry @ Rz. Simulation only; no collision checking.
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
    printer = scenery.printer1
    kind = part.kind
    sizes = {'barrel': (0.050, 0.240), 'body': (0.07560, 0.17498)}
    width, height = sizes[kind]

    root = tk.Tk()
    root.title(f'IR-A2 | {kind.upper()} | Pick pose teach GUI')
    root.geometry('690x780')
    ttk.Label(root, text=f'Part: {kind} | Pick pose teach pendant',
              font=('Segoe UI', 13, 'bold')).pack(pady=(12, 4))
    ttk.Label(root, text='XYZ = STL-local mm | Rx/Ry/Rz = printer plate_reference degrees',
              foreground='#075985').pack()
    ttk.Label(root, text='Simulation only. Direct jtraj is NOT collision checked.',
              foreground='#9b4200').pack(pady=(2, 8))

    status = tk.StringVar(value='Ready. Run existing pipeline, then preview IK.')
    values = {}
    labels = {}
    q_goal = {'value': None}
    prepared = {'value': False}

    def invalidate():
        q_goal['value'] = None
        status.set('Pose changed. Run IK Preview again.')

    def update_position(*_):
        if not all(a in values for a in ('X', 'Y', 'Z')):
            return
        xyz = [values[a].get() / 1000.0 for a in ('X', 'Y', 'Z')]
        marker.T = (part.pose @ SE3(*xyz)).A
        for axis in ('X', 'Y', 'Z'):
            labels[axis].set(f'{values[axis].get():.1f}')
        invalidate()
        env.step(0.01)

    def update_orientation(*_):
        if not all(a in values for a in ('Rx', 'Ry', 'Rz')):
            return
        for axis in ('Rx', 'Ry', 'Rz'):
            labels[axis].set(f'{values[axis].get():.1f}')
        invalidate()

    def slider(parent, axis, lo, hi, default, unit, callback):
        row = ttk.Frame(parent)
        row.pack(fill='x', padx=14, pady=6)
        ttk.Label(row, text=axis, width=7).pack(side='left')
        var = tk.DoubleVar(value=default)
        values[axis] = var
        ttk.Scale(row, from_=lo, to=hi, variable=var, command=callback).pack(
            side='left', fill='x', expand=True, padx=8)
        label = tk.StringVar(value=f'{default:.1f}')
        labels[axis] = label
        ttk.Label(row, textvariable=label, width=8).pack(side='left')
        ttk.Label(row, text=unit, width=5).pack(side='left')

    position_box = ttk.LabelFrame(root, text='Pick marker position (STL local)')
    position_box.pack(fill='x', padx=14, pady=6)
    slider(position_box, 'X', -30, 120, width * 500, 'mm', update_position)
    slider(position_box, 'Y', -80, 120, 0, 'mm', update_position)
    slider(position_box, 'Z', 0, 270, height * 2000 / 3, 'mm', update_position)

    rotation_box = ttk.LabelFrame(root, text='Target EE orientation (plate frame)')
    rotation_box.pack(fill='x', padx=14, pady=6)
    slider(rotation_box, 'Rx', -180, 180, -90, 'deg', update_orientation)
    slider(rotation_box, 'Ry', -180, 180, 0, 'deg', update_orientation)
    slider(rotation_box, 'Rz', -180, 180, 0, 'deg', update_orientation)

    def target_pose():
        p = np.asarray(marker.T, dtype=float)[:3, 3]
        rx, ry, rz = [np.deg2rad(values[a].get()) for a in ('Rx', 'Ry', 'Rz')]
        R_plate = printer.plate_reference.R
        R = R_plate @ SE3.Rx(rx).R @ SE3.Ry(ry).R @ SE3.Rz(rz).R
        return SE3.Rt(R, p)

    def prep_robot():
        if prepared['value']:
            messagebox.showinfo('Already prepared', 'Restart GUI to rerun door opening.')
            return
        if not messagebox.askyesno('Run existing pipeline',
                'Open door → retreat → clearance → standby → camera → pre-pick?\n'
                'This runs existing RobotTasks without changing retreat orientation.\n'
                'No automatic collision checking.'):
            return
        try:
            tasks.open_door(printer)
            tasks.retract_from_door(printer)
            tasks.move_to_clearance(printer)
            tasks.reorient_for_standby(printer)
            tasks.return_to_standby(printer)
            tasks.move_to_camera_pose(printer)
            tasks.move_to_pre_pick(printer, distance=0.10)
            prepared['value'] = True
            invalidate()
            status.set('Pre-pick reached. Set Rx/Ry/Rz and preview IK.')
        except Exception as exc:
            messagebox.showerror('Pipeline failed', str(exc))

    def preview_ik():
        if not prepared['value']:
            messagebox.showwarning('Not ready', 'Run the existing door → pre-pick pipeline first.')
            return
        try:
            T = target_pose()
            sol = robot.ikine_LM(T, q0=robot.q.copy(), joint_limits=True)
            if not sol.success:
                raise RuntimeError('IK did not converge. Adjust XYZ or Rx/Ry/Rz.')
            actual = robot.fkine(sol.q)
            pos_mm = float(np.linalg.norm(actual.t - T.t) * 1000)
            angle = float(np.degrees(np.arccos(np.clip(
                (np.trace(T.R.T @ actual.R) - 1) / 2, -1, 1))))
            q_goal['value'] = np.asarray(sol.q, dtype=float).copy()
            status.set(f'IK SUCCESS | XYZ error {pos_mm:.2f} mm | R error {angle:.3f} deg')
            print('Target WORLD XYZ (m):', np.round(T.t, 5))
            print('Target plate-relative Rx/Ry/Rz (deg):',
                  [round(values[a].get(), 2) for a in ('Rx', 'Ry', 'Rz')])
            print('IK joint target (deg):', np.round(np.rad2deg(sol.q), 2))
            print('IK errors: mm', round(pos_mm, 3), 'deg', round(angle, 4))
        except Exception as exc:
            q_goal['value'] = None
            messagebox.showerror('IK preview failed', str(exc))

    def move_robot():
        if q_goal['value'] is None:
            messagebox.showwarning('No IK', 'Run IK Preview after your latest slider change.')
            return
        if not messagebox.askyesno('Direct joint trajectory — NO collision checking',
                'Move directly to the target XYZ + Rx/Ry/Rz using jtraj?\n'
                'This can pass through printer walls, door, or part.\n'
                'Simulation only; visually inspect intersections.'):
            return
        try:
            path = jtraj(robot.q.copy(), q_goal['value'], 180).q
            for q in path[1:]:
                robot.q = np.asarray(q, dtype=float)
                env.step(0.025)
            q_goal['value'] = None
            status.set('Direct jtraj completed. Check geometry for collisions.')
        except Exception as exc:
            messagebox.showerror('Motion failed', str(exc))

    ttk.Button(root, text='1. Run existing door → camera → pre-pick',
               command=prep_robot).pack(fill='x', padx=16, pady=5)
    ttk.Button(root, text='2. IK Preview (position + orientation; no motion)',
               command=preview_ik).pack(fill='x', padx=16, pady=5)
    ttk.Button(root, text='3. TEST direct jtraj (no collision checking)',
               command=move_robot).pack(fill='x', padx=16, pady=5)
    ttk.Label(root, textvariable=status, wraplength=630,
              foreground='#075985').pack(padx=16, pady=12)
    ttk.Label(root, text='Set Scenery.part1_type to barrel or body. The 3 orientation sliders control the IK target, not the retreat motion.',
              wraplength=630).pack(padx=16, pady=3)
    update_position()
    update_orientation()
    root.mainloop()


if __name__ == '__main__':
    main()
