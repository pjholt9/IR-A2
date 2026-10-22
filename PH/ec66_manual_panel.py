"""EC66 + linear rail manual pendant for the integrated IR-A2 workcell.

Only the main simulation thread calls pump().  This panel intentionally does not
use Swift/PyBullet collision flags; it enforces joint/rail bounds, not geometric
collision clearance.  Never operate it while an automatic cycle is running.
"""
import json
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
from spatialmath import SE3
from .ec66_collision_check import EC66CollisionChecker


MAX_JOINT_STEP = np.deg2rad(1.0)  # per main-loop pump
MAX_RAIL_STEP = 0.003  # m per pump
EPS = 1e-5
WAYPOINT_FILE = Path(__file__).resolve().parent.parent / 'Do' / 'manual_ec66_waypoint.json'


class EC66ManualPanel:
    def __init__(self, root, scenery, safety, on_close=None):
        self.root, self.scenery, self.safety = root, scenery, safety
        self.on_close = on_close
        self.window = None
        self.returning = False
        self.q_target = None
        self.rail_target = None
        self._home_q = np.asarray(scenery.ec66.q_standby, dtype=float).copy()
        self._rail_stations = [0.0]
        for printer in scenery.printers:
            station = scenery.get_printer_station(printer)
            rel = scenery.platform.home_pose.inv() @ station
            self._rail_stations.append(float(rel.t[0]))
        self.rail_min = min(self._rail_stations)
        self.rail_max = max(self._rail_stations)
        self.message = None
        self.checker = EC66CollisionChecker(scenery)
        self.collision_latched = False
        self.collision_reason = None

    @property
    def active(self):
        return self.window is not None or self.returning

    def _rail_position(self):
        rel = self.scenery.platform.home_pose.inv() @ self.scenery.platform.pose
        return float(rel.t[0])

    def open(self):
        if self.window is not None:
            self.window.lift()
            return
        if self.returning:
            return
        win = self.window = tk.Toplevel(self.root)
        win.title('EC66 + Rail | Manual Teach')
        win.geometry('515x600')
        win.protocol('WM_DELETE_WINDOW', self.close)
        ttk.Label(win, text='EC66 + Linear Rail | Manual Teach', font=('Arial', 13, 'bold')).pack(pady=10)
        ttk.Label(win, text='Approximate line–box collision guard (not safety-rated)', foreground='darkorange').pack()
        self.joint_vars = []
        self.joint_labels = []
        robot = self.scenery.robot
        for i in range(robot.n):
            row = ttk.Frame(win)
            row.pack(fill='x', padx=10, pady=3)
            ttk.Label(row, text=f'J{i+1}', width=4).pack(side='left')
            value = tk.DoubleVar(value=float(np.rad2deg(robot.q[i])))
            lo, hi = np.rad2deg(robot.qlim[:, i])
            ttk.Scale(row, from_=float(lo), to=float(hi), variable=value).pack(side='left', fill='x', expand=True)
            label = ttk.Label(row, text='', width=11)
            label.pack(side='right')
            self.joint_vars.append(value)
            self.joint_labels.append(label)
        rail = ttk.LabelFrame(win, text='Linear rail (relative to home, metres)')
        rail.pack(fill='x', padx=10, pady=9)
        rail_row = ttk.Frame(rail)
        rail_row.pack(fill='x', padx=6, pady=5)
        self.rail_var = tk.DoubleVar(value=self._rail_position())
        ttk.Scale(rail_row, from_=self.rail_min, to=self.rail_max, variable=self.rail_var).pack(side='left', fill='x', expand=True)
        self.rail_label = ttk.Label(rail_row, width=11)
        self.rail_label.pack(side='right')
        self.pose_label = ttk.Label(win, text='')
        self.pose_label.pack(pady=4)
        self.message = ttk.Label(win, text='Ready to jog', wraplength=480)
        self.message.pack(pady=5)
        self.collision_label = ttk.Label(win, text='No collision', foreground='green', font=('Arial', 11, 'bold'))
        self.collision_label.pack(pady=3)
        ttk.Button(win, text='Acknowledge collision / enable recovery jog', command=self.ack_collision).pack(pady=3)
        controls = ttk.Frame(win)
        controls.pack(fill='x', padx=10, pady=5)
        ttk.Button(controls, text='Capture waypoint JSON', command=self.capture).pack(fill='x', pady=3)
        ttk.Button(controls, text='Load saved waypoint', command=self.load_waypoint).pack(fill='x', pady=3)
        ttk.Button(controls, text='Sliders to standby + rail home', command=self.sliders_home).pack(fill='x', pady=3)
        ttk.Button(controls, text='Exit manual (return to standby + home)', command=self.close).pack(fill='x', pady=3)
        self.q_target = np.asarray(robot.q, dtype=float).copy()
        self.rail_target = self._rail_position()

    def ack_collision(self):
        if self.safety.stopped:
            return
        self.collision_latched = False
        self.collision_reason = None
        if self.window is not None:
            self.collision_label.config(text='No collision — recovery jog enabled', foreground='green')
            self.message.config(text='Choose a clear direction; future collisions remain blocked')

    def sliders_home(self):
        for var, angle in zip(self.joint_vars, np.rad2deg(self._home_q)):
            var.set(float(angle))
        self.rail_var.set(0.0)

    def capture(self):
        if self.window is None:
            return
        q = np.asarray(self.scenery.robot.q, dtype=float).copy()
        rail = self._rail_position()
        ee = self.scenery.robot.fkine(q).t
        data = {'robot': 'EC66', 'q_deg': np.round(np.rad2deg(q), 6).tolist(),
                'rail_from_home_m': round(rail, 6),
                'ee_world_xyz_m': np.round(ee, 6).tolist(),
                'saved_at': datetime.now().isoformat(timespec='seconds'),
                'note': 'Manual teaching only; no geometric collision verification'}
        WAYPOINT_FILE.write_text(json.dumps(data, indent=2), encoding='utf-8')
        self.message.config(text=f'SAVED current robot pose: {WAYPOINT_FILE.name}')
        print('EC66 MANUAL WAYPOINT SAVED:', data)

    def load_waypoint(self):
        try:
            data = json.loads(WAYPOINT_FILE.read_text(encoding='utf-8'))
            q = np.asarray(data['q_deg'], dtype=float)
            rail = float(data['rail_from_home_m'])
            if q.shape != (self.scenery.robot.n,) or not np.all(np.isfinite(q)):
                raise ValueError('Invalid joint vector')
            limits = np.rad2deg(self.scenery.robot.qlim)
            if np.any(q < limits[0]) or np.any(q > limits[1]) or not self.rail_min <= rail <= self.rail_max:
                raise ValueError('Waypoint outside joint or rail limits')
            for var, angle in zip(self.joint_vars, q):
                var.set(float(angle))
            self.rail_var.set(rail)
            self.message.config(text='Loaded waypoint targets (jogging toward them)')
        except (OSError, KeyError, ValueError, TypeError) as exc:
            self.message.config(text=f'Load failed: {exc}')

    def close(self):
        if self.window is None:
            return
        # Do not close the teach panel if the planned homing motion is unsafe.
        # The operator must recover to a clear pose before exiting manual mode.
        q0 = np.asarray(self.scenery.robot.q, dtype=float).copy()
        r0 = self._rail_position()
        hit = self.checker.swept_check(q0, r0, self._home_q, 0.0, samples=65)
        if hit:
            self.collision_latched = True
            self.collision_reason = hit
            self.collision_label.config(text=f'HOME BLOCKED: {hit}', foreground='red')
            self.message.config(text='Acknowledge and jog to a clear pose before exiting.')
            return
        self.window.destroy()
        self.window = None
        self.returning = True
        self.collision_latched = False
        self.q_target = self._home_q.copy()
        self.rail_target = 0.0

    def _move_rail(self, position):
        pose = self.scenery.platform.get_station_pose(position)
        self.scenery.platform.update_pose(pose)
        self.scenery.ec66.base = self.scenery.platform.pose @ self.scenery.robot_offset
        self.scenery.ec66.update_base()

    def pump(self):
        if not self.active or self.safety.stopped:
            return
        robot = self.scenery.robot
        if self.window is not None:
            self.q_target = np.deg2rad([var.get() for var in self.joint_vars])
            self.rail_target = float(self.rail_var.get())
        limits = np.asarray(robot.qlim, dtype=float)
        q_target = np.clip(self.q_target, limits[0], limits[1])
        q_now = np.asarray(robot.q, dtype=float)
        q_next = np.clip(q_now + np.clip(q_target - q_now, -MAX_JOINT_STEP, MAX_JOINT_STEP), limits[0], limits[1])
        rail_now = self._rail_position()
        rail_target = float(np.clip(self.rail_target, self.rail_min, self.rail_max))
        rail_next = rail_now + float(np.clip(rail_target - rail_now, -MAX_RAIL_STEP, MAX_RAIL_STEP))
        # Predict collisions BEFORE applying the joint/rail update. The check
        # temporarily changes only robot.base and restores it immediately.
        if self.collision_latched and not self.returning:
            return
        hit = self.checker.swept_check(q_now, rail_now, q_next, rail_next)
        if hit:
            self.collision_latched = True
            self.collision_reason = hit
            if self.window is not None:
                self.collision_label.config(text=f'COLLISION DETECTED: {hit}', foreground='red')
                self.message.config(text='Motion blocked. Acknowledge, then jog away from the obstacle.')
                # Reset requested targets to actual safe pose so it will not
                # immediately repeat the same unsafe motion after acknowledgement.
                for var, angle in zip(self.joint_vars, np.rad2deg(q_now)):
                    var.set(float(angle))
                self.rail_var.set(rail_now)
            if self.returning:
                # Do not unlock Automatic if homing unexpectedly fails.
                print('EC66 RETURN BLOCKED BY COLLISION:', hit)
            return
        # Robot and platform updated together, before the next Swift frame.
        self._move_rail(rail_next)
        robot.q = q_next
        if self.returning:
            if np.max(np.abs(q_next - self._home_q)) < EPS and abs(rail_next) < EPS:
                robot.q = self._home_q.copy()
                self._move_rail(0.0)
                self.returning = False
                if self.on_close:
                    self.on_close()
            return
        for label, angle in zip(self.joint_labels, np.rad2deg(q_next)):
            label.config(text=f'{angle:+.1f}°')
        self.rail_label.config(text=f'{rail_next:.3f} m')
        ee = robot.fkine(q_next).t
        self.pose_label.config(text=f'EE world: X={ee[0]:.3f}  Y={ee[1]:.3f}  Z={ee[2]:.3f} m')
        if not self.collision_latched:
            self.collision_label.config(text='No collision', foreground='green')
