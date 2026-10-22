"""Manual TX2-60 joint control and gripper, as a window inside the combined workcell GUI.

Unlike manual_control.py this has no loop of its own: pump() is called from the main
simulation thread (every Swift step) so the automatic cells keep running around it.
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
from spatialgeometry import Sphere
from spatialmath import SE3

from .collision import in_collision
from .scenery_PH import PART_RADIUS, TABLE_TOP

# Keep manual commands within the workcell and move gradually between slider targets.
PARKED = (100.0, 100.0, -100.0)
XY_RANGE = 0.95  # m either side of the robot base (about the arm's reach)
Z_ABOVE_BASE = 0.95
Z_MIN_ABOVE_TABLE = 0.03  # keep the gripper off the table surface
MAX_JOINT_STEP = np.deg2rad(3.0)  # per pump, so a slider jump becomes a smooth move
GRIP_RANGE = 0.10  # m, end-effector to part centre
FALL_STEP = 0.03  # m per pump when a released part drops


# Manual controls share the main simulation loop, so the automated cell can keep running alongside them.
class ManualPanel:
    # Set up state once; the panel window and its widgets are created only when manual mode opens.
    def __init__(self, root, tx2_60, env, general_bucket_pose, on_close=None):
        self.root = root
        self.tx2_60 = tx2_60
        self.on_close = on_close
        self.rest_z = float(general_bucket_pose.t[2])
        self.z_min = float(TABLE_TOP + Z_MIN_ABOVE_TABLE)
        self.general_pose = SE3(general_bucket_pose)
        self.window = None
        self.returning = False
        self.falling = False
        self.q_target = None
        self.part = Sphere(radius=PART_RADIUS, color="orange", pose=SE3(*PARKED))
        self.part._collision = False
        env.add(self.part)

    # The panel remains active while its close animation returns the arm to standby.
    @property
    def active(self):
        return self.window is not None or self.returning

    # Build the sliders, Cartesian controls and gripper buttons if the panel is not already open.
    def open(self):
        if self.window is not None:
            self.window.lift()
            return
        self.returning = False
        robot = self.tx2_60.robot
        qlim = robot.qlim
        win = self.window = tk.Toplevel(self.root)
        win.title("TX2-60 manual control")
        win.protocol("WM_DELETE_WINDOW", self.close)

        self.sliders = []
        self.value_labels = []
        for i in range(robot.n):
            row = ttk.Frame(win, padding=(8, 3))
            row.pack(fill="x")
            ttk.Label(row, text=f"q{i + 1}", width=3).pack(side="left")
            var = tk.DoubleVar(value=float(np.rad2deg(robot.q[i])))
            ttk.Scale(row, from_=float(np.rad2deg(qlim[0, i])), to=float(np.rad2deg(qlim[1, i])),
                      variable=var, length=300).pack(side="left", padx=6)
            label = ttk.Label(row, text="", width=8)
            label.pack(side="left")
            self.sliders.append(var)
            self.value_labels.append(label)

        # Show the live tool pose so joint moves can be checked in Cartesian space.
        self.pose_label = ttk.Label(win, text="", padding=4)
        self.pose_label.pack()

        # Cartesian control: dragging X/Y/Z solves IK (tool pointing down) and drives the joint sliders
        xyz_frame = ttk.LabelFrame(win, text="End-effector position (world, m)")
        xyz_frame.pack(fill="x", padx=8, pady=4)
        base = self.tx2_60.base.t
        ranges = (
            (base[0] - XY_RANGE, base[0] + XY_RANGE),
            (base[1] - XY_RANGE, base[1] + XY_RANGE),
            (self.z_min, base[2] + Z_ABOVE_BASE),
        )
        ee = self.tx2_60.get_ee_pose().t
        # While dragging, IK below drives the same joint sliders used by joint-space control.
        self.xyz_vars = []
        self.xyz_dragging = False
        for name, (lo, hi), value in zip("XYZ", ranges, ee):
            row = ttk.Frame(xyz_frame, padding=(4, 2))
            row.pack(fill="x")
            ttk.Label(row, text=name, width=2).pack(side="left")
            var = tk.DoubleVar(value=float(value))
            scale = ttk.Scale(row, from_=float(lo), to=float(hi), variable=var, length=300,
                              command=lambda _v: self.on_xyz_change())
            scale.pack(side="left", padx=6)
            scale.bind("<ButtonPress-1>", lambda _e: setattr(self, "xyz_dragging", True))
            scale.bind("<ButtonRelease-1>", lambda _e: setattr(self, "xyz_dragging", False))
            self.xyz_vars.append(var)
        self.xyz_status = ttk.Label(xyz_frame, text="Drag to move the gripper; the arm stays within its reach")
        self.xyz_status.pack(pady=(0, 4))
        self.collision_label = ttk.Label(win, text="")
        self.collision_label.pack()
        self.grip_label = ttk.Label(win, text="", padding=4)
        self.grip_label.pack()

        buttons = ttk.Frame(win, padding=8)
        buttons.pack(fill="x")
        self.grip_button = ttk.Button(buttons, text="Grip part", command=self.toggle_grip)
        self.grip_button.pack(fill="x", pady=2)
        ttk.Button(buttons, text="Put test part in general bin", command=self.place_part).pack(fill="x", pady=2)
        ttk.Button(buttons, text="Sliders to standby", command=self.sliders_to_standby).pack(fill="x", pady=2)
        ttk.Button(buttons, text="Exit manual mode (return to standby)", command=self.close).pack(fill="x", pady=2)
        self._set_grip_text()
        self.last_safe = np.asarray(robot.q, dtype=float).copy()
        # One-shot diagnostics: show whether the workcell collision objects are registered.
        obs = self.tx2_60.obstacles
        print(f"[TX2 COLLISION DEBUG] Manual opened | obstacles={len(obs)}", flush=True)
        for i, obstacle in enumerate(obs):
            try:
                centre = np.asarray(obstacle.T, dtype=float)[:3, 3]
                scale = getattr(obstacle, "scale", "unknown")
                print(f"  [{i:02d}] {type(obstacle).__name__} center={np.round(centre, 3)} scale={scale}", flush=True)
            except Exception as exc:
                print(f"  [{i:02d}] {type(obstacle).__name__} inspection failed: {exc}", flush=True)

    # Convert the requested XYZ point to a downward-facing tool pose, then let IK check reachability.
    def on_xyz_change(self):
        """Solve IK for the requested position; reject it if unreachable or outside joint limits."""
        if not self.xyz_dragging:
            return
        robot = self.tx2_60.robot
        target = np.array([var.get() for var in self.xyz_vars])
        pose = self.tx2_60._downward_pose(SE3(*target))
        q0 = np.deg2rad([var.get() for var in self.sliders])
        sol = robot.ikine_LM(pose, q0=q0)
        if sol.success:
            q = self.tx2_60._nearest_q(sol.q)
            qlim = robot.qlim
            ok = bool(np.all(q >= qlim[0] - 1e-6) and np.all(q <= qlim[1] + 1e-6))
        else:
            ok = False
        if ok:
            for var, angle in zip(self.sliders, np.rad2deg(q)):
                var.set(float(angle))
            self.xyz_status.config(text="Target reachable", foreground="green")
        else:
            self.xyz_status.config(text="Out of reach / joint limits: arm not moved", foreground="red")

    # Closing the window doesn't teleport the robot; pump() eases it back to the standby joints.
    def close(self):
        """Release anything held, hide the test part, and let pump() walk the arm back to standby."""
        if self.window is None:
            return
        self.window.destroy()
        self.window = None
        self.tx2_60.attached_object = None
        self.falling = False
        self._move_part(SE3(*PARKED))
        self.returning = True

    # Fill the joint targets with the known safe standby configuration.
    def sliders_to_standby(self):
        for var, q in zip(self.sliders, self.tx2_60.q_standby):
            var.set(float(np.rad2deg(q)))

    # Drop the orange test part straight into the general bucket for a quick GUI demo.
    def place_part(self):
        self.tx2_60.attached_object = None
        self.falling = False
        self._move_part(self.general_pose)
        self._set_grip_text()
        self.grip_label.config(text="Test part placed in the general bin")

    # Attach only when the gripper is close enough; otherwise leave the part where it is.
    def toggle_grip(self):
        tx2_60 = self.tx2_60
        if tx2_60.attached_object is not None:
            tx2_60.attached_object = None
            self.falling = True
            self.grip_label.config(text="Released")
        else:
            ee = tx2_60.get_ee_pose()
            part_pose = SE3(np.asarray(self.part.T, dtype=float))
            if np.linalg.norm(ee.t - part_pose.t) > GRIP_RANGE:
                self.grip_label.config(text="Part out of reach: move the gripper onto the orange part")
            else:
                tx2_60.attached_object = self.part
                tx2_60.grasp_offset = ee.inv() * part_pose
                self.falling = False
                self.grip_label.config(text="Gripped")
        self._set_grip_text()

    # Keep the button label matched to whether the robot currently holds the part.
    def _set_grip_text(self):
        holding = self.tx2_60.attached_object is not None
        self.grip_button.config(text="Release part" if holding else "Grip part")

    # Move the visual test part by assigning its homogeneous transform.
    def _move_part(self, pose):
        self.part.T = SE3(pose).A

    # Called once per simulation step: clamp motion, check collision, and update the GUI and test part.
    def pump(self):
        """Advance manual control one step. Main thread only."""
        tx2_60 = self.tx2_60
        robot = tx2_60.robot
        if self.window is None and not self.returning:
            return
        if self.returning:
            step = np.clip(tx2_60.q_standby - robot.q, -MAX_JOINT_STEP, MAX_JOINT_STEP)
            robot.q = robot.q + step
            # Snap tiny final errors away so the return state can finish cleanly.
            if np.allclose(robot.q, tx2_60.q_standby, atol=1e-6):
                robot.q = np.asarray(tx2_60.q_standby, dtype=float).copy()
                self.returning = False
                if self.on_close:
                    self.on_close()
            return

        target = np.deg2rad([var.get() for var in self.sliders])
        q = robot.q + np.clip(target - robot.q, -MAX_JOINT_STEP, MAX_JOINT_STEP)
        # Collision rejection keeps the robot at the last pose known to be clear.
        hit = bool(tx2_60.obstacles) and in_collision(robot, q, tx2_60.obstacles)
        if hit and not getattr(self, "_last_collision_debug", False):
            print(f"[TX2 COLLISION DEBUG] BLOCKED | obstacles={len(tx2_60.obstacles)} | q_deg={np.round(np.rad2deg(q), 1)}", flush=True)
        self._last_collision_debug = bool(hit)
        if hit:
            # Refuse the move and put the sliders back at the last safe pose
            q = self.last_safe.copy()
            for var, angle in zip(self.sliders, q):
                var.set(float(np.rad2deg(angle)))
        else:
            self.last_safe = q.copy()
        robot.q = q

        # A held part follows the tool; a released one gets a simple stepwise drop to the bin height.
        if tx2_60.attached_object is not None:
            tx2_60.attached_object.T = (tx2_60.get_ee_pose() @ tx2_60.grasp_offset).A
        elif self.falling:
            T = np.array(self.part.T, dtype=float)
            T[2, 3] = max(self.rest_z, T[2, 3] - FALL_STEP)
            self.part.T = T
            if T[2, 3] <= self.rest_z:
                self.falling = False

        for label, angle in zip(self.value_labels, np.rad2deg(q)):
            label.config(text=f"{angle:7.1f}°")
        xyz = tx2_60.get_ee_pose().t
        self.pose_label.config(text=f"End-effector: x={xyz[0]:.3f}  y={xyz[1]:.3f}  z={xyz[2]:.3f} m")
        if not self.xyz_dragging:
            for var, value in zip(self.xyz_vars, xyz):
                var.set(float(value))
        self.collision_label.config(
            text="COLLISION - move blocked" if hit else "No collision",
            foreground="red" if hit else "green",
        )
