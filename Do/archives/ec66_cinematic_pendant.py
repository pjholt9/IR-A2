"""EC66 cinematic joint teach pendant. Place beside original scenery.py in Do/."""
import tkinter as tk
from tkinter import ttk
import numpy as np
import swift
from scenery import Scenery

DT = 0.025
MAX_SPEED_DEG_S = 35.0  # smooth cinematic motion


def main():
    env = swift.Swift()
    env.launch(realtime=True)
    scenery = Scenery()
    scenery.add_to_env(env)  # original project method adds component meshes
    robot = scenery.robot
    if robot is None:
        raise RuntimeError("Restore original scenery.py with EC66 and add_to_env().")
    robot.q = scenery.ec66.q_standby.copy()
    env.step(DT)

    limits = np.asarray([link.qlim for link in robot.links], dtype=float)
    if limits.shape != (6, 2):
        raise RuntimeError(f"Expected six EC66 joints, got limits {limits.shape}")
    limits_deg = np.rad2deg(limits)
    initial = np.rad2deg(robot.q.copy())

    root = tk.Tk()
    root.title("EC66 | Cinematic Teach Pendant")
    root.geometry("570x560")
    root.configure(bg="#15191f")
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TButton", padding=7)
    tk.Label(root, text="EC66  /  CINEMATIC CONTROL", fg="white", bg="#15191f",
             font=("Segoe UI", 17, "bold")).pack(pady=(17, 5))
    tk.Label(root, text="Joint-space motion • smooth speed-limited tracking",
             fg="#aeb6c2", bg="#15191f", font=("Segoe UI", 10)).pack(pady=(0, 14))

    target_vars = []
    actual_labels = []
    for j in range(6):
        frame = tk.Frame(root, bg="#15191f")
        frame.pack(fill="x", padx=18, pady=5)
        tk.Label(frame, text=f"J{j+1}", width=4, fg="white", bg="#15191f",
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        low, high = limits_deg[j]
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = -180.0, 180.0
        v = tk.DoubleVar(value=float(initial[j]))
        target_vars.append(v)
        scale = tk.Scale(frame, from_=float(low), to=float(high), resolution=0.5,
                         variable=v, orient="horizontal", length=350,
                         showvalue=True, fg="#e9edf4", bg="#15191f",
                         troughcolor="#394657", highlightthickness=0,
                         activebackground="#55b6ff")
        scale.pack(side="left", fill="x", expand=True)
        label = tk.Label(frame, text=f"{initial[j]:6.1f}°", width=9,
                         fg="#7fd6a6", bg="#15191f", font=("Consolas", 10))
        label.pack(side="right")
        actual_labels.append(label)

    options = tk.Frame(root, bg="#15191f")
    options.pack(fill="x", padx=22, pady=12)
    speed = tk.DoubleVar(value=18.0)
    tk.Label(options, text="Max speed (°/s)", fg="white", bg="#15191f").pack(side="left")
    ttk.Scale(options, from_=2, to=MAX_SPEED_DEG_S, variable=speed, length=190).pack(side="left", padx=12)
    speed_label = tk.Label(options, text="18.0", fg="white", bg="#15191f")
    speed_label.pack(side="left")

    paused = tk.BooleanVar(value=False)
    def set_target(q):
        for v, angle in zip(target_vars, np.rad2deg(q)):
            v.set(float(angle))

    def reset():
        set_target(scenery.ec66.q_standby)

    controls = tk.Frame(root, bg="#15191f")
    controls.pack(pady=4)
    ttk.Button(controls, text="Standby", command=reset).pack(side="left", padx=5)
    ttk.Button(controls, text="Hold position", command=lambda: set_target(robot.q.copy())).pack(side="left", padx=5)
    ttk.Checkbutton(controls, text="Pause", variable=paused).pack(side="left", padx=10)
    status = tk.Label(root, text="Ready • adjust sliders to move joints", fg="#aeb6c2", bg="#15191f")
    status.pack(pady=12)

    running = True
    def close():
        nonlocal running
        running = False
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", close)

    def tick():
        if not running:
            return
        try:
            speed_label.config(text=f"{speed.get():.1f}")
            if not paused.get():
                q_target = np.deg2rad([v.get() for v in target_vars])
                q_current = robot.q.copy()
                max_step = np.deg2rad(speed.get()) * DT
                dq = np.clip(q_target - q_current, -max_step, max_step)
                q_next = np.clip(q_current + dq, limits[:, 0], limits[:, 1])
                robot.q = q_next
            env.step(DT)
            for j, label in enumerate(actual_labels):
                label.config(text=f"{np.rad2deg(robot.q[j]):6.1f}°")
        except Exception as exc:
            status.config(text=f"Simulation error: {exc}", fg="#ff8c8c")
            paused.set(True)
        root.after(int(DT * 1000), tick)

    root.after(50, tick)
    root.mainloop()


if __name__ == "__main__":
    main()
