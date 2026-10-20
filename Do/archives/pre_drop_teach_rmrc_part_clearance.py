"""Pre-drop teach GUI for combined Do + PH scenery.

Run from IR-A2 root: .\\.venv\\Scripts\\python.exe Do\\pre_drop_teach.py
Uses the validated RobotTasks pick sequence directly. No core files modified.
No collision detection. All slider positions are WORLD XYZ in mm.
"""
import argparse
import json
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import swift
from spatialgeometry import Sphere
from spatialmath import SE3
from roboticstoolbox import jtraj

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from scenery import Scenery
from Do.robot_tasks_backup import RobotTasks
from PH.scenery_PH import SceneryPH, BUCKET_WIDTH, BUCKET_WALL_HEIGHT, PART_RADIUS

SAVE_PATH = HERE / 'pre_drop_waypoint.json'

# Release height above bucket floor (millimetres), selected by part kind.
PART_RELEASE_CLEARANCE_MM = {'barrel': 50.0, 'body': 80.0}



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--printer', type=int, default=1, choices=[1, 2])
    args = ap.parse_args()
    env = swift.Swift()
    env.launch(realtime=True)
    ph = SceneryPH()
    ph.add_to_env(env)
    scene = Scenery()
    scene.add_to_env(env)
    tasks = RobotTasks(scene, env)
    printer = scene.printers[args.printer - 1]
    robot = scene.robot
    part = scene.parts.get(printer)
    if part is None:
        raise RuntimeError('No part selected in this printer')
    if part.kind not in PART_RELEASE_CLEARANCE_MM:
        raise ValueError(f'No release clearance configured for part kind: {part.kind}')
    bucket = ph.general_bucket_pose
    # PH bucket pose = part resting centre, NOT top rim or bottom floor.
    # PH/_add_bucket: floor centre = bucket @ SE3(0,0,-PART_RADIUS-FLOOR_THICKNESS/2)
    # Therefore the INSIDE floor top is bucket.z - PART_RADIUS.
    floor_z = float(bucket.t[2] - PART_RADIUS)
    rim_z = float(floor_z + BUCKET_WALL_HEIGHT)
    print('General bucket resting-centre WORLD XYZ:', bucket.t)
    print('Bucket inside floor WORLD Z:', floor_z)
    print('Bucket top rim WORLD Z:', rim_z)
    print('NOTE: PH has a separate sorting robot; it remains stationary here.')

    # Visual reference only; not a collision obstacle.
    target_marker = Sphere(radius=0.014, pose=SE3(float(bucket.t[0]), float(bucket.t[1]), rim_z + 0.15), color=[0, 0.8, 1, 1])
    target_marker._collision = False
    env.add(target_marker)
    state = {'prepared': False, 'q_hold': None, 'plan': None, 'done': False,
             'recorder': None, 'preview_start': None, 'drop_plan': None, 'drop_start': None, 'dropped': False}

    root = tk.Tk()
    root.title('IR-A2 | Pre-drop teach | EC66 + PH workcell')
    root.geometry('850x850')
    ttk.Label(root, text='Pre-drop waypoint teaching', font=('Segoe UI', 16, 'bold')).pack(pady=12)
    ttk.Label(root, text=f'World XYZ (mm), world RPY (degrees). {part.kind.upper()} default clearance: {PART_RELEASE_CLEARANCE_MM[part.kind]:.0f} mm above bucket floor.').pack()
    ttk.Label(root, text='Pre-drop uses jtraj; drop uses Jacobian DLS RMRC. NO collision checking.', foreground='#aa5500').pack(pady=6)
    status = tk.StringVar(value='Run existing pick to 600 mm retreat, then teach pre-drop.')
    sliders = {}
    labels = {}
    T0 = SE3(float(bucket.t[0]), float(bucket.t[1]), rim_z + 0.15)
    defaults = {'X': T0.t[0]*1000, 'Y': T0.t[1]*1000, 'Z': T0.t[2]*1000,
                'Rx': -90., 'Ry': 0., 'Rz': 0., 'Clearance': PART_RELEASE_CLEARANCE_MM[part.kind]}
    limits = {'X': (-1500, 1500), 'Y': (-1000, 1800), 'Z': (500, 1800),
              'Rx': (-180, 180), 'Ry': (-180, 180), 'Rz': (-180, 180), 'Clearance': (0, 120)}
    if SAVE_PATH.exists():
        try:
            saved = json.loads(SAVE_PATH.read_text())
            # World waypoint shared between parts, clearance is per-part.
            defaults.update({k: float(saved[k]) for k in defaults if k != 'Clearance' and k in saved})
            per_part = saved.get('ClearanceByPart', {})
            if isinstance(per_part, dict) and part.kind in per_part:
                defaults['Clearance'] = float(per_part[part.kind])
            # Legacy single Clearance intentionally ignored: it may belong to another part.
        except (ValueError, OSError):
            pass

    def save_waypoint():
        # Keep previous part's clearance if switching model later.
        saved = {}
        if SAVE_PATH.exists():
            try:
                saved = json.loads(SAVE_PATH.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                pass
        per_part = saved.get('ClearanceByPart', {})
        if not isinstance(per_part, dict):
            per_part = {}
        per_part[part.kind] = round(sliders['Clearance'].get(), 4)
        values = {k: round(v.get(), 4) for k, v in sliders.items() if k != 'Clearance'}
        values['ClearanceByPart'] = per_part
        SAVE_PATH.write_text(json.dumps(values, indent=2), encoding='utf-8')

    def invalidate(*_):
        state['plan'] = None
        state['drop_plan'] = None
        if not state['done']:
            status.set('Waypoint changed. Preview IK again.')
        try:
            target_marker.T = pose_from_sliders()
        except (KeyError, ValueError):
            pass

    for name in defaults:
        line = ttk.Frame(root)
        line.pack(fill='x', padx=22, pady=6)
        ttk.Label(line, text=name, width=9).pack(side='left')
        var = tk.DoubleVar(value=defaults[name])
        sliders[name] = var
        ttk.Scale(line, from_=limits[name][0], to=limits[name][1], variable=var,
                  command=invalidate).pack(side='left', fill='x', expand=True, padx=8)
        lab = tk.StringVar(value=f'{defaults[name]:.1f}')
        labels[name] = lab
        ttk.Label(line, textvariable=lab, width=12).pack(side='left')
        var.trace_add('write', lambda *_a, k=name, v=var: labels[k].set(f'{v.get():.1f}'))
    def pose_from_sliders():
        r, p, y = [np.deg2rad(sliders[k].get()) for k in ('Rx', 'Ry', 'Rz')]
        return SE3(*(sliders[k].get()/1000 for k in ('X', 'Y', 'Z'))) @ SE3.Rx(r) @ SE3.Ry(p) @ SE3.Rz(y)

    def prepare():
        if state['prepared']:
            messagebox.showinfo('Already prepared', 'Pick and 600 mm retreat already completed.')
            return
        try:
            # Exact sequence from the latest ik_test_modular.py.
            recorder = tasks.open_door(printer)
            tasks.retract_from_door(printer)
            tasks.move_to_clearance(printer)
            tasks.reorient_for_standby(printer)
            tasks.return_to_standby(printer)
            tasks.move_to_camera_pose(printer)
            tasks.camera_capture_dwell(printer, seconds=3.0)
            tasks.move_to_pre_pick(printer, distance=0.10)
            pick_plan = tasks.plan_part_pick(printer, part)
            success = tasks.execute_part_pick(printer, part, pick_plan)
            if not success or not part.isAttached:
                raise RuntimeError('ATTACH FAILED: pick stopped; pre-drop disabled')
            state['recorder'] = recorder
            state['q_hold'] = np.asarray(robot.q, dtype=float).copy()
            state['prepared'] = True
            state['plan'] = None
            state['done'] = False
            state['dropped'] = False
            status.set('Pick + retreat 600 mm complete. Teach and preview pre-drop.')
            print('READY TO TEACH | attached:', part.isAttached,
                  '| EE world XYZ:', np.round(robot.fkine(robot.q).t, 5))
        except Exception as exc:
            status.set('Preparation failed; inspect terminal.')
            messagebox.showerror('Prepare failed', str(exc))

    def preview():
        if not state['prepared']:
            messagebox.showwarning('Not prepared', 'Prepare the pick + 600mm retreat first.')
            return
        try:
            target = pose_from_sliders()
            sol = robot.ikine_LM(target, q0=state['q_hold'], joint_limits=True)
            if not sol.success:
                raise RuntimeError('Pre-drop IK failed')
            # Check the drop target above the box rim (EE, not part bottom).
            if target.t[2] <= rim_z:
                raise RuntimeError('Pre-drop EE target is below the bucket rim')
            q_end = np.asarray(sol.q, dtype=float)
            qlim = np.asarray([link.qlim for link in robot.links], dtype=float)
            if np.any(q_end < qlim[:, 0]) or np.any(q_end > qlim[:, 1]):
                raise RuntimeError('IK target violates joint limits')
            state['plan'] = q_end
            state['preview_start'] = np.asarray(robot.q, dtype=float).copy()
            err = np.linalg.norm(robot.fkine(q_end).t - target.t) * 1000
            status.set(f'IK preview OK | EE position error {err:.2f} mm. Visual collision check required.')
        except Exception as exc:
            state['plan'] = None
            messagebox.showerror('IK preview', str(exc))

    def step_motion(q):
        robot.q = np.asarray(q, dtype=float).copy()
        if part.isAttached:
            part.follow_ee()  # Original, tested attachment transform.
        env.step(0.03)
        if state['recorder'] is not None:
            state['recorder'].record(0.03, 'Pre-drop teach')
        root.update_idletasks()

    def move_to_pre_drop():
        if state['plan'] is None or not state['prepared']:
            messagebox.showwarning('Preview first', 'Preview the target from the 600mm retreat pose first.')
            return
        try:
            if np.max(np.abs(robot.q - state['preview_start'])) > 1e-5:
                raise RuntimeError('Robot moved after preview; preview again.')
            for q in jtraj(robot.q.copy(), state['plan'], 240).q:
                step_motion(q)
            state['drop_plan'] = None
            status.set('Reached pre-drop. Preview calculated vertical RMRC next.')
            save_waypoint()
            state['done'] = True
        except Exception as exc:
            messagebox.showerror('Motion failed', str(exc))

    def part_bottom_geometry(T_mesh):
        """Lowest footprint corner in WORLD Z; robust to small part tilt.

        Mesh.T is the STL-local-to-world transform in the current PrintedPart
        implementation. STL local coordinates are metres after mesh scale.
        """
        from printed_part import PART_CONFIG
        config = PART_CONFIG[part.kind]
        w, d = float(config['width']), float(config['depth'])
        corners = np.array([[0., 0., 0., 1.], [w, 0., 0., 1.],
                            [0., d, 0., 1.], [w, d, 0., 1.]]).T
        xyz = (np.asarray(T_mesh, dtype=float) @ corners)[:3, :]
        return float(xyz[2].min()), xyz

    def calculate_drop():
        if not state['done'] or not state['prepared'] or state['dropped']:
            raise RuntimeError('Run pick and move to pre-drop before drop preview.')
        if not part.isAttached:
            raise RuntimeError('Part must still be attached to EE.')
        T_ee = robot.fkine(robot.q)
        T_mesh = np.asarray(part.mesh.T, dtype=float)
        z_min, corners = part_bottom_geometry(T_mesh)
        clearance = float(sliders['Clearance'].get()) / 1000.0
        if not np.isfinite(clearance) or clearance < 0:
            raise ValueError('Release clearance must be nonnegative and finite')
        travel = z_min - floor_z - clearance
        if not np.isfinite(travel) or travel <= 0:
            raise RuntimeError(f'No downward clearance: bottom Z={z_min:.4f}, floor={floor_z:.4f}')
        if travel > 0.40:
            raise RuntimeError(f'Drop travel {travel*1000:.1f} mm > 400 mm safety limit.')
        # All bottom corners should fit inside the bucket opening at final pose.
        inner_half = BUCKET_WIDTH / 2 - 0.02  # 20mm wall thickness
        for j in range(corners.shape[1]):
            if (abs(corners[0, j] - bucket.t[0]) > inner_half or
                    abs(corners[1, j] - bucket.t[1]) > inner_half):
                raise RuntimeError('Part footprint outside bucket inner opening.')
        target = SE3.Rt(T_ee.R.copy(), T_ee.t + np.array([0., 0., -travel]))
        return travel, z_min, clearance, target

    def preview_drop():
        try:
            travel, z_min, clearance, T_goal = calculate_drop()
            q0 = np.asarray(robot.q, dtype=float).copy()
            # Reuse the EXACT Jacobian DLS RMRC planner validated for picking.
            path_down = tasks._plan_linear_pick_rmrc(
                q0, T_goal, steps=max(120, int(np.ceil(travel / 0.0015))))
            # Retrace the identical joint path backwards. This guarantees the
            # same geometric trajectory without introducing another IK branch.
            path_up = path_down[::-1].copy()
            state['drop_plan'] = {'down': path_down, 'up': path_up,
                                  'distance': travel, 'clearance': clearance,
                                  'z_bottom': z_min, 'T_goal': T_goal}
            state['drop_start'] = q0
            status.set(f'Drop preview OK | down/up {travel*1000:.1f} mm | '
                       f'part bottom {z_min:.3f} m | floor {floor_z:.3f} m')
            print('\n========== GEOMETRY-BASED DROP PREVIEW ==========')
            print('Part:', part.kind, '| STL height:', part.height, 'm')
            print('EE pre-drop XYZ:', np.round(robot.fkine(q0).t, 5))
            print('Part lowest bottom Z:', round(z_min, 5))
            print('PH bucket floor top Z:', round(floor_z, 5))
            print('Release clearance (mm):', round(clearance * 1000, 2))
            print('Computed down/up travel (mm):', round(travel * 1000, 2))
            print('Target EE XYZ:', np.round(T_goal.t, 5))
            print('No motion or detach executed yet.')
        except Exception as exc:
            state['drop_plan'] = None
            messagebox.showerror('Drop RMRC preview failed', str(exc))

    def execute_drop():
        plan = state['drop_plan']
        if plan is None:
            messagebox.showwarning('Preview first', 'Preview the calculated drop RMRC first.')
            return
        if state['dropped'] or not part.isAttached:
            messagebox.showwarning('Already dropped', 'Part is no longer attached.')
            return
        if np.max(np.abs(np.asarray(robot.q) - state['drop_start'])) > 1e-5:
            messagebox.showerror('Robot moved', 'Robot moved since preview. Preview again.')
            return
        if not messagebox.askyesno('Execute simulated drop',
                f"Move DOWN {plan['distance']*1000:.1f} mm, detach {part.kind}, "
                f"then retract UP the same distance?\n\n"
                'Simulation only; NO geometry collision checking.'):
            return
        try:
            for q in plan['down'][1:]:
                step_motion(q)
            # Release the part at its real carried pose. Then move it straight
            # down by the small clearance to rest on the bucket floor.
            # This is a simplified gravity-settle, NOT a rigid-body simulation.
            part.detach()
            T_mesh = np.asarray(part.mesh.T, dtype=float).copy()
            z_min_after, _ = part_bottom_geometry(T_mesh)
            settle = z_min_after - floor_z
            T_mesh[:3, 3] -= np.array([0., 0., settle])
            part.mesh.T = T_mesh
            print(f'DETACH SUCCESS | isAttached={part.isAttached} | '
                  f'part settled {settle*1000:.2f} mm to bucket floor')
            state['dropped'] = True
            for q in plan['up'][1:]:
                step_motion(q)
            state['drop_plan'] = None
            status.set(f'DROP COMPLETE | {part.kind} on bucket floor | '
                       f'RMRC down/up {plan["distance"]*1000:.1f} mm | EE held above bucket')
            print('DROP COMPLETE: RMRC return to pre-drop; no standby motion.')
        except Exception as exc:
            state['drop_plan'] = None
            status.set('Drop interrupted. Inspect robot and part before retry.')
            messagebox.showerror('Drop execution interrupted', str(exc))

    ttk.Button(root, text='1. Run validated door → camera → pick → retreat 600 mm', command=prepare).pack(fill='x', padx=20, pady=7)
    ttk.Button(root, text='2. Preview pre-drop IK (no motion)', command=preview).pack(fill='x', padx=20, pady=7)
    ttk.Button(root, text='3. Move to pre-drop (jtraj, teach only)', command=move_to_pre_drop).pack(fill='x', padx=20, pady=7)
    ttk.Button(root, text='4. PREVIEW calculated drop RMRC (no motion)', command=preview_drop).pack(fill='x', padx=20, pady=7)
    ttk.Button(root, text='5. TEST RMRC down → detach → reverse RMRC up', command=execute_drop).pack(fill='x', padx=20, pady=7)
    ttk.Label(root, textvariable=status, foreground='#116699', wraplength=770).pack(pady=12)
    root.mainloop()


if __name__ == '__main__':
    main()
