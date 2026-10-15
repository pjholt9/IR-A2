"""EC66 A2: compare rail positions and pre-pre-grasp waypoints.

Run from repository root:
    python Do/ik_test.py

IMPORTANT: Set station2 in linear_platform.py to REFERENCE_RAIL_X (0.85 m).
RAIL_AXIS_WORLD must match the actual direction of increasing station2 distance.
Visual clearance is NOT an automated collision check.
"""

from math import pi

import matplotlib.pyplot as plt
import numpy as np
import swift
from roboticstoolbox import jtraj
from scipy import linalg
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from scenery import Scenery


# ========================== CONFIGURATION ==========================
REFERENCE_RAIL_X = 0.85  # Current station2 value in linear_platform.py (m)
RAIL_AXIS_WORLD = np.array([1.0, 0.0, 0.0])  # Verify direction in Swift!

CANDIDATES = {
    0.80: np.array([-0.15, -0.03, 0.15]),
    0.85: np.array([-0.03, 0.00, 0.10]),
    0.90: np.array([-0.06, -0.05, 0.30]),
}

ANIMATE_COMPARISON = False  # Fast calculation, no per-step Swift animation
ANIMATE_WINNER = True       # Replay best valid candidate in Swift
PLOT_RESULTS = True

DT_JTRAJ = 0.02
N_JTRAJ = 200
DT_RMRC = 0.03
N_RMRC = 150
RMRC_DAMPING = 0.001
KP = 2.0
KR = 2.0
DT_DOOR = 0.05
DOOR_ANGLES = np.linspace(0.0, -pi / 2, 91)
JOINT_JUMP_LIMIT_DEG = 10.0
POSITION_TOLERANCE = 0.005  # m
ORIENTATION_TOLERANCE = np.deg2rad(2.0)


class Recorder:
    def __init__(self, q0):
        self.time = [0.0]
        self.q = [np.asarray(q0, dtype=float).copy()]
        self.phase = ["Standby"]

    def add(self, q, dt, phase):
        self.time.append(self.time[-1] + dt)
        self.q.append(np.asarray(q, dtype=float).copy())
        self.phase.append(phase)


def pose_error(target, actual):
    pos = np.linalg.norm(target.t - actual.t)
    rot = np.linalg.norm(Rotation.from_matrix(target.R @ actual.R.T).as_rotvec())
    return pos, rot


def within_limits(q, qlim, tolerance=1e-8):
    return bool(np.all(q >= qlim[:, 0] - tolerance) and
                np.all(q <= qlim[:, 1] + tolerance))


def ik_or_fail(robot, target, seed, name):
    sol = robot.ikine_LM(target, q0=seed.copy(), joint_limits=True)
    if not sol.success:
        raise RuntimeError(f"IK failed: {name}")
    pos, rot = pose_error(target, robot.fkine(sol.q))
    if pos > POSITION_TOLERANCE or rot > ORIENTATION_TOLERANCE:
        raise RuntimeError(f"IK inaccurate: {name} ({pos*1000:.2f} mm, "
                           f"{np.rad2deg(rot):.2f} deg)")
    return sol.q.copy()


def record_jtraj(robot, recorder, q_target, phase, animate, env):
    q_start = robot.q.copy()
    traj = jtraj(q_start, q_target, N_JTRAJ).q
    # Skip the duplicate first point (zero elapsed time).
    for q in traj[1:]:
        robot.q = q.copy()
        recorder.add(robot.q, DT_JTRAJ, phase)
        if animate:
            env.step(DT_JTRAJ)


def run_candidate(scenery, env, rail_x, offset, reference_pose, animate=False):
    robot = scenery.robot
    door = scenery.printer1.door
    qlim = np.asarray([link.qlim for link in robot.links], dtype=float)

    # station2 is calibrated at REFERENCE_RAIL_X. Shift in WORLD coordinates.
    delta = (rail_x - REFERENCE_RAIL_X) * RAIL_AXIS_WORLD
    target_pose = SE3.Rt(reference_pose.R, reference_pose.t + delta)
    if animate:
        scenery.move_platform(target_pose, env, steps=120)
    else:
        scenery.platform.update_pose(target_pose)
        scenery.ec66.base = scenery.platform.pose @ scenery.robot_offset
        scenery.ec66.update_base()

    robot.q = scenery.ec66.q_standby.copy()
    door.update_angle(0.0)
    recorder = Recorder(robot.q)

    t_contact = door.get_handle_pose() @ SE3.Rx(-pi / 2)
    t_pregrasp = t_contact @ SE3(0, 0, -0.10)
    candidate_xyz = t_pregrasp.t + offset  # Offset in WORLD XYZ.
    t_candidate = SE3.Rt(t_pregrasp.R, candidate_xyz)

    # 1. Standby -> pre-pre-grasp -> pre-grasp.
    q_candidate = ik_or_fail(robot, t_candidate, robot.q, "Pre-pre-grasp")
    record_jtraj(robot, recorder, q_candidate, "Pre-pre-grasp", animate, env)
    q_pregrasp = ik_or_fail(robot, t_pregrasp, robot.q, "Pre-grasp")
    record_jtraj(robot, recorder, q_pregrasp, "Pre-grasp", animate, env)

    # 2. Full-pose, closed-loop RMRC to handle.
    start_xyz = robot.fkine(robot.q).t.copy()
    positions = np.linspace(start_xyz, t_contact.t, N_RMRC)
    for i in range(N_RMRC - 1):
        actual = robot.fkine(robot.q)
        desired_xyz = positions[i + 1]
        linear = ((positions[i + 1] - positions[i]) / DT_RMRC
                  + KP * (desired_xyz - actual.t))
        angular = KR * Rotation.from_matrix(
            t_contact.R @ actual.R.T
        ).as_rotvec()
        twist = np.r_[linear, angular]
        J = robot.jacob0(robot.q)
        qdot = J.T @ linalg.solve(
            J @ J.T + RMRC_DAMPING * np.eye(6), twist,
            assume_a="pos"
        )
        q_next = robot.q + DT_RMRC * qdot
        if not within_limits(q_next, qlim):
            raise RuntimeError(f"RMRC joint limit at step {i}")
        robot.q = q_next
        recorder.add(robot.q, DT_RMRC, "Approach")
        if animate:
            env.step(DT_RMRC)

    contact_pos_err, contact_rot_err = pose_error(t_contact, robot.fkine(robot.q))
    if (contact_pos_err > POSITION_TOLERANCE or
            contact_rot_err > ORIENTATION_TOLERANCE):
        raise RuntimeError(f"Handle tracking error: {contact_pos_err*1000:.2f} mm, "
                           f"{np.rad2deg(contact_rot_err):.2f} deg")

    # 3. Continuous IK door opening; seed each solve from previous q.
    q_open = [robot.q.copy()]  # Exactly preserve the RMRC end configuration.
    try:
        for angle in DOOR_ANGLES[1:]:
            door.update_angle(angle)
            target = door.get_handle_pose() @ SE3.Rx(-pi / 2)
            q_next = ik_or_fail(robot, target, q_open[-1],
                                f"Door {np.rad2deg(angle):.1f} deg")
            jump = np.max(np.abs(np.rad2deg(q_next - q_open[-1])))
            if jump > JOINT_JUMP_LIMIT_DEG:
                raise RuntimeError(f"Door IK jump {jump:.2f} deg")
            q_open.append(q_next)
    finally:
        door.update_angle(0.0)

    for angle, q in zip(DOOR_ANGLES[1:], q_open[1:]):
        door.update_angle(angle)
        robot.q = q.copy()
        recorder.add(robot.q, DT_DOOR, "Door opening")
        if animate:
            env.step(DT_DOOR)

    return {
        "rail_x": rail_x,
        "offset": offset.copy(),
        "candidate_xyz": candidate_xyz.copy(),
        "recorder": recorder,
        "contact_error_mm": contact_pos_err * 1000,
        "contact_error_deg": np.rad2deg(contact_rot_err),
    }


def calculate_metrics(robot, result):
    rec = result["recorder"]
    time = np.asarray(rec.time)
    q = np.asarray(rec.q)
    phases = np.asarray(rec.phase)
    qlim = np.asarray([link.qlim for link in robot.links])
    J = [robot.jacob0(qi) for qi in q]
    singular = np.asarray([np.linalg.svd(j, compute_uv=False) for j in J])
    # Yoshikawa 6D manipulability (same full Jacobian as original code).
    manip = np.prod(singular, axis=1)
    sigma_min = singular[:, -1]
    margins = np.minimum(q - qlim[:, 0], qlim[:, 1] - q)
    qdot_deg = np.rad2deg(np.diff(q, axis=0) / np.diff(time)[:, None])

    # Standby is a possible wrist singularity; report task-specific results.
    task_mask = np.isin(phases, ["Approach", "Door opening"])
    # Also report full moving trajectory without the initial standby sample.
    moving_mask = phases != "Standby"
    metrics = {
        "min_sigma_task": float(np.min(sigma_min[task_mask])),
        "median_sigma_task": float(np.median(sigma_min[task_mask])),
        "min_manip_task": float(np.min(manip[task_mask])),
        "median_manip_task": float(np.median(manip[task_mask])),
        "min_sigma_moving": float(np.min(sigma_min[moving_mask])),
        "min_j3_margin_deg": float(np.rad2deg(np.min(margins[moving_mask, 2]))),
        "min_all_margin_deg": float(np.rad2deg(np.min(margins[moving_mask]))),
        "max_joint_speed_deg_s": float(np.max(np.abs(qdot_deg))),
        "limit_violation": not within_limits(q.min(axis=0), qlim) or
                           not within_limits(q.max(axis=0), qlim),
        "duration_s": float(time[-1]),
    }
    result.update({"time": time, "q": q, "phases": phases,
                   "manip": manip, "sigma_min": sigma_min,
                   "j3_margin_deg": np.rad2deg(margins[:, 2]),
                   "qdot_deg": qdot_deg, "metrics": metrics})
    return result


def print_comparison(results):
    print("\n========== EC66 RAIL COMPARISON ==========")
    print("All Jacobian metrics use the full 6D Jacobian.")
    print("Task = Approach + Door opening; excludes Standby/transfer phases.")
    print("Visual clearance is NOT a verified collision check.\n")
    header = (f"{'Rail':>6} {'Min sigma':>12} {'Median manip':>14} "
              f"{'J3 margin':>11} {'Max speed':>11} {'Status':>10}")
    print(header)
    print("-" * len(header))
    for x, result in sorted(results.items()):
        if "error" in result:
            print(f"{x:6.2f} {'-':>12} {'-':>14} {'-':>11} {'-':>11} {'FAILED':>10}")
            print(f"       Reason: {result['error']}")
            continue
        m = result["metrics"]
        status = "LIMIT" if m["limit_violation"] else "OK*"
        print(f"{x:6.2f} {m['min_sigma_task']:12.6f} "
              f"{m['median_manip_task']:14.6f} "
              f"{m['min_j3_margin_deg']:11.2f} "
              f"{m['max_joint_speed_deg_s']:11.2f} {status:>10}")
    print("* OK means IK/tracking/joint-limit checks passed, NOT collision-free.")
    print("==========================================\n")


def plot_comparison(results):
    valid = [r for r in results.values() if "error" not in r]
    if not valid:
        return
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
    for r in valid:
        label = f"Rail {r['rail_x']:.2f} m"
        t = r["time"]
        axes[0].plot(t, r["sigma_min"], label=label)
        axes[1].plot(t, r["manip"], label=label)
        axes[2].plot(t, r["j3_margin_deg"], label=label)
    for ax, ylabel in zip(axes, ["Minimum singular value (6D)",
                                   "Manipulability (6D)",
                                   "Joint 3 limit margin (deg)"]):
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Motion time (s)")
        ax.grid(True)
        ax.legend()
    fig.suptitle("EC66 - Rail position comparison (all phases)")
    fig.tight_layout()
    plt.show()


def main():
    env = swift.Swift()
    env.launch(realtime=True)
    scenery = Scenery()
    scenery.add_to_env(env)
    env.step(0.1)

    reference_pose = scenery.platform.station2.copy()
    results = {}
    for x, offset in CANDIDATES.items():
        print(f"\nEvaluating rail {x:.2f} m, waypoint offset {offset} ...", flush=True)
        try:
            result = run_candidate(scenery, env, x, offset, reference_pose,
                                   animate=ANIMATE_COMPARISON)
            results[x] = calculate_metrics(scenery.robot, result)
            print(f"  OK | handle error {result['contact_error_mm']:.3f} mm, "
                  f"{result['contact_error_deg']:.4f} deg")
        except (RuntimeError, ValueError, linalg.LinAlgError) as exc:
            results[x] = {"error": str(exc)}
            print(f"  FAILED: {exc}")

    print_comparison(results)
    valid = [r for r in results.values() if "error" not in r and
             not r["metrics"]["limit_violation"]]
    if valid:
        # Primary: best worst-case task singular value. Tie-break: J3 margin.
        winner = max(valid, key=lambda r: (r["metrics"]["min_sigma_task"],
                                           r["metrics"]["min_j3_margin_deg"]))
        print(f"Kinematic front-runner: rail {winner['rail_x']:.2f} m "
              f"(not collision-validated).")
        if ANIMATE_WINNER:
            print("Replaying front-runner in Swift...")
            run_candidate(scenery, env, winner["rail_x"], winner["offset"],
                          reference_pose, animate=True)
    else:
        print("No candidate completed all kinematic checks.")

    if PLOT_RESULTS:
        plot_comparison(results)
    while True:
        env.step(0.05)


if __name__ == "__main__":
    main()
