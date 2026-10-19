"""Tkinter window with one slider per TX2-60 joint, for driving the arm by hand in Swift."""

import tkinter as tk
from tkinter import ttk

import numpy as np

if __package__:
    from .collision import in_collision
else:
    from collision import in_collision


def run_manual_control(tx2_60, env, step_dt=0.02):
    """Open the joint-angle GUI and step Swift until the window is closed.

    Sliders are limited to the DH model's qlim. A move that would put a link inside a
    registered obstacle is refused and the sliders snap back to the last safe pose.
    """
    robot = tx2_60.robot
    qlim = robot.qlim
    root = tk.Tk()
    root.title("TX2-60 manual joint control")

    sliders = []
    value_labels = []
    for i in range(robot.n):
        row = ttk.Frame(root, padding=(8, 4))
        row.pack(fill="x")
        ttk.Label(row, text=f"q{i + 1}", width=3).pack(side="left")
        var = tk.DoubleVar(value=float(np.rad2deg(robot.q[i])))
        ttk.Scale(
            row,
            from_=float(np.rad2deg(qlim[0, i])),
            to=float(np.rad2deg(qlim[1, i])),
            variable=var,
            length=320,
        ).pack(side="left", padx=6)
        label = ttk.Label(row, text="", width=8)
        label.pack(side="left")
        sliders.append(var)
        value_labels.append(label)

    pose_label = ttk.Label(root, text="", padding=8)
    pose_label.pack()
    collision_label = ttk.Label(root, text="", padding=(8, 0, 8, 8))
    collision_label.pack()

    def reset():
        for var, q in zip(sliders, tx2_60.q_standby):
            var.set(float(np.rad2deg(q)))

    ttk.Button(root, text="Reset to standby", command=reset).pack(pady=(0, 8))

    running = {"on": True}
    root.protocol("WM_DELETE_WINDOW", lambda: running.update(on=False))

    last_safe = np.asarray(robot.q, dtype=float).copy()
    while running["on"]:
        q = np.deg2rad([var.get() for var in sliders])
        hit = bool(tx2_60.obstacles) and in_collision(robot, q, tx2_60.obstacles)
        if hit:
            # Refuse the move: snap the sliders back to the last collision-free pose
            q = last_safe.copy()
            for var, angle in zip(sliders, q):
                var.set(float(np.rad2deg(angle)))
        else:
            last_safe = q.copy()
        robot.q = q
        if tx2_60.attached_object is not None:
            tx2_60.attached_object.T = tx2_60.get_ee_pose() @ tx2_60.grasp_offset
        for label, angle in zip(value_labels, np.rad2deg(q)):
            label.config(text=f"{angle:7.1f}°")
        xyz = tx2_60.get_ee_pose().t
        pose_label.config(text=f"End-effector: x={xyz[0]:.3f}  y={xyz[1]:.3f}  z={xyz[2]:.3f} m")
        collision_label.config(
            text="COLLISION - move blocked" if hit else "No collision",
            foreground="red" if hit else "green",
        )
        env.step(step_dt)
        root.update()
    root.destroy()
