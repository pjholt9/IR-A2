"""Teach/test return from standby to OPEN door handle, including a synchronized reverse door-close test.

Place beside scenery.py and robot_tasks.py in Do/.
Run from repository root: .\\.venv\\Scripts\\python.exe .\\Do\\door_handle_return_teach.py

Uses actual recorded joint paths for reverse reorientation, clearance and retract.
No geometry collision checking; verify visually in Swift.
"""
import tkinter as tk
from tkinter import messagebox
import traceback
import numpy as np
import swift
from spatialmath import SE3
from roboticstoolbox import jtraj
from scipy.spatial.transform import Rotation
from scenery import Scenery
from Do.robot_tasks_backup import RobotTasks


def main():
    env = swift.Swift()
    env.launch(realtime=True)
    scene = Scenery()
    scene.add_to_env(env)
    env.step(0.1)
    robot = scene.robot
    printer = scene.printer1
    tasks = RobotTasks(scene, env)
    state = {'stage': 0, 'paths': {}, 'recorder': None}

    root = tk.Tk()
    root.title('IR-A2 | Door handle return TEACH - diagnostic v3')
    root.geometry('700x480')
    status = tk.StringVar(value='Ready. Start by recording the validated opening path.')

    def record_segment(label, operation):
        rec = state['recorder']
        q_before = robot.q.copy()
        start_idx = len(rec.q)
        operation()
        samples = [q_before] + [np.asarray(q, dtype=float).copy() for q in rec.q[start_idx:]]
        if len(samples) < 2:
            raise RuntimeError(f'No recorded joint samples for {label}')
        samples[-1] = robot.q.copy()  # authoritative final joint state
        state['paths'][label] = np.asarray(samples).copy()
        print(f'{label}: saved {len(samples)} joint samples; final q(deg) =',
              np.round(np.rad2deg(robot.q), 2))

    def replay_reverse(label):
        path = state['paths'][label]
        mismatch_vector = np.rad2deg(np.asarray(robot.q) - path[-1])
        mismatch = np.max(np.abs(mismatch_vector))
        print(f'{label} reverse start | actual q(deg):', np.round(np.rad2deg(robot.q), 2))
        print(f'{label} reverse start | saved q(deg): ', np.round(np.rad2deg(path[-1]), 2))
        print(f'{label} reverse start | delta(deg):   ', np.round(mismatch_vector, 2))
        if mismatch > 2.0:
            raise RuntimeError(f'{label}: start mismatch {mismatch:.2f} deg (>2 deg)')
        for q in path[-2::-1]:
            robot.q = q.copy()
            env.step(0.03)
            state['recorder'].record(0.03, 'Reverse ' + label)
        print(f'Reverse {label}: EE XYZ = {np.round(robot.fkine(robot.q).t, 5)}')

    def move_to_reoriented_clearance():
        """Move from standby to the EXACT saved end of reorientation."""
        print("\n========== STEP 2 ENTERED ==========", flush=True)
        if 'reorient' not in state['paths']:
            raise RuntimeError('No recorded reorientation path. Run PREPARE first.')

        q_start = np.asarray(robot.q, dtype=float).copy().reshape(-1)
        q_goal = np.asarray(state['paths']['reorient'][-1], dtype=float).copy().reshape(-1)
        if q_start.shape != q_goal.shape:
            raise RuntimeError(f'Joint vector shape mismatch: {q_start.shape} vs {q_goal.shape}')

        print('Start q (deg): ', np.round(np.rad2deg(q_start), 2), flush=True)
        print('Target q (deg):', np.round(np.rad2deg(q_goal), 2), flush=True)

        # RTB robot.qlim is shaped (2, n): first row lower limits,
        # second row upper limits. Do NOT stack link.qlim as (n, 2)
        # and then index it as (2, n).
        limits = np.asarray(robot.qlim, dtype=float)
        print('Joint limits shape:', limits.shape, flush=True)
        if limits.shape != (2, len(q_start)):
            raise RuntimeError(f'Unexpected robot.qlim shape {limits.shape}; expected (2, {len(q_start)})')

        trajectory = np.asarray(jtraj(q_start, q_goal, 220).q, dtype=float)
        if trajectory.shape != (220, len(q_start)):
            raise RuntimeError(f'Unexpected trajectory shape {trajectory.shape}')

        # Joint limits are checked only where finite; a continuous joint
        # may have +/-inf limits in some robot models.
        lower, upper = limits[0], limits[1]
        tol = 1e-6
        violation = (np.isfinite(lower) & (trajectory < lower - tol)) | (np.isfinite(upper) & (trajectory > upper + tol))
        if np.any(violation):
            step_idx, joint_idx = np.argwhere(violation)[0]
            raise RuntimeError(
                f'Step 2 joint limit violation at sample {step_idx}, joint {joint_idx + 1}: '
                f'{np.rad2deg(trajectory[step_idx, joint_idx]):.2f} deg '
                f'(allowed {np.rad2deg(lower[joint_idx]):.2f} to '
                f'{np.rad2deg(upper[joint_idx]):.2f} deg). '
                'Do not bypass this check; inspect the saved joint branch.'
            )

        print('Step 2: joint-limit check passed. Starting jtraj...', flush=True)
        for i, q in enumerate(trajectory[1:], start=1):
            robot.q = q.copy()
            env.step(0.02)
            state['recorder'].record(0.02, 'Standby to clearance')
            if i % 55 == 0 or i == len(trajectory) - 1:
                print(f'Step 2 progress: {i}/{len(trajectory)-1}', flush=True)

        actual = np.asarray(robot.q, dtype=float)
        error_deg = float(np.rad2deg(np.max(np.abs(actual - q_goal))))
        print(f'Standby -> clearance endpoint error: {error_deg:.4f} deg', flush=True)
        print('Actual q (deg):', np.round(np.rad2deg(actual), 2), flush=True)
        print('Target q (deg):', np.round(np.rad2deg(q_goal), 2), flush=True)
        if error_deg > 2.0:
            raise RuntimeError(f'Step 2 endpoint mismatch {error_deg:.2f} deg; cannot reverse reorientation')
        print('========== STEP 2 FINISHED ==========\n', flush=True)

    def check_handle():
        target = printer.door.get_handle_pose() @ SE3.Rx(-np.pi / 2)
        actual = robot.fkine(robot.q)
        err_mm = 1000 * np.linalg.norm(actual.t - target.t)
        err_deg = np.rad2deg(np.linalg.norm(
            Rotation.from_matrix(target.R @ actual.R.T).as_rotvec()))
        print(f'OPEN HANDLE REACHED | XYZ error {err_mm:.3f} mm | R error {err_deg:.4f} deg')
        if err_mm > 5.0 or err_deg > 2.0:
            raise RuntimeError('Handle pose mismatch; do not start door closing')

    def close_door_reverse():
        """Replay the verified opening joint path backwards with the door angle.

        This is reverse trajectory playback, NOT a new IK or Jacobian RMRC solve.
        The opening code samples 91 angles [0, -pi/2] and records 90 moves.
        """
        rec = state['recorder']
        opening_indices = [i for i, phase in enumerate(rec.phase)
                           if phase == 'Door opening']
        if len(opening_indices) != 90:
            raise RuntimeError(
                f'Expected 90 door-opening samples, got {len(opening_indices)}. '
                'Opening trajectory may have changed; aborting.')
        if opening_indices != list(range(opening_indices[0], opening_indices[0] + 90)):
            raise RuntimeError('Door-opening samples are not contiguous; aborting.')

        # The preceding recorder sample is the RMRC arrival at the CLOSED handle.
        start_idx = opening_indices[0] - 1
        if start_idx < 0:
            raise RuntimeError('Missing closed-handle joint sample')
        opening_q = np.asarray([rec.q[start_idx]] +
                               [rec.q[i] for i in opening_indices], dtype=float)
        opening_angles = np.linspace(0.0, -np.pi / 2, 91)
        if opening_q.shape != (91, len(robot.q)):
            raise RuntimeError(f'Unexpected opening trajectory shape {opening_q.shape}')

        q_error = float(np.rad2deg(np.max(np.abs(robot.q - opening_q[-1]))))
        print(f'CLOSE START | joint mismatch {q_error:.6f} deg', flush=True)
        if q_error > 2.0:
            raise RuntimeError(f'Not at open handle: joint mismatch {q_error:.3f} deg')

        closed_handle = printer.door.get_handle_pose()  # door is still OPEN
        _ = closed_handle  # pose checked again at each replay step
        worst_position_mm = 0.0
        worst_orientation_deg = 0.0
        for i in range(89, -1, -1):
            # Exactly reverse the joint samples AND the corresponding door angle.
            printer.door.update_angle(float(opening_angles[i]))
            robot.q = opening_q[i].copy()
            env.step(0.05)
            rec.record(0.05, 'Door closing (reverse playback)')

            target = printer.door.get_handle_pose() @ SE3.Rx(-np.pi / 2)
            actual = robot.fkine(robot.q)
            pos_mm = 1000.0 * float(np.linalg.norm(actual.t - target.t))
            rot_deg = float(np.rad2deg(np.linalg.norm(
                Rotation.from_matrix(target.R @ actual.R.T).as_rotvec())))
            worst_position_mm = max(worst_position_mm, pos_mm)
            worst_orientation_deg = max(worst_orientation_deg, rot_deg)
            if i % 15 == 0:
                print(f'CLOSING | angle={np.rad2deg(opening_angles[i]):.1f} deg | '
                      f'handle error={pos_mm:.3f} mm, {rot_deg:.4f} deg', flush=True)
            if pos_mm > 5.0 or rot_deg > 2.0:
                raise RuntimeError(
                    f'Door closing alignment exceeded tolerance at angle '
                    f'{np.rad2deg(opening_angles[i]):.1f} deg: '
                    f'{pos_mm:.3f} mm / {rot_deg:.3f} deg. Motion stopped.')

        print('DOOR CLOSED | worst handle error '
              f'{worst_position_mm:.3f} mm / {worst_orientation_deg:.4f} deg', flush=True)
        print('EE remains at CLOSED handle; no release/standby motion performed.', flush=True)

    def prepare():
        rec = tasks.open_door(printer)
        state['recorder'] = rec
        record_segment('retract', lambda: tasks.retract_from_door(printer))
        record_segment('clearance', lambda: tasks.move_to_clearance(printer))
        record_segment('reorient', lambda: tasks.reorient_for_standby(printer))
        tasks.return_to_standby(printer)
        state['stage'] = 1
        print('PREPARE COMPLETE | stage=2 | robot q(deg):', np.round(np.rad2deg(robot.q), 2), flush=True)
        status.set('Door OPEN, robot at standby. Ready for reverse approach.')

    steps = [
        ('1. PREPARE: open door → retract → clearance → reorient → standby', prepare),
        ('2. Standby → taught clearance waypoint (jtraj)', move_to_reoriented_clearance),
        ('3. Reverse reorientation at clearance', lambda: replay_reverse('reorient')),
        ('4. Reverse clearance → retract pose', lambda: replay_reverse('clearance')),
        ('5. Reverse retract → OPEN handle',
         lambda: (replay_reverse('retract'), check_handle())),
        ('6. CLOSE DOOR: reverse verified opening trajectory', close_door_reverse),
    ]

    def execute(index):
        print(f"GUI CLICK | requested step={index + 1} | stage={state['stage'] + 1} | "
              f"q(deg)={np.round(np.rad2deg(robot.q), 2)}", flush=True)

        # A previous step 2 may have been marked complete while the robot
        # remained at standby. Permit re-running it until the actual joint
        # configuration matches the recorded reorientation endpoint.
        if index == 1 and state['stage'] == 2:
            target = state['paths'].get('reorient')
            if target is not None:
                mismatch = float(np.max(np.abs(np.rad2deg(
                    np.asarray(robot.q) - np.asarray(target[-1])))))
                if mismatch > 2.0:
                    print(f"Step 2 RETRY allowed: target mismatch {mismatch:.2f} deg", flush=True)
                    state['stage'] = 1

        if index != state['stage']:
            messagebox.showwarning('Sequence',
                f'Run step {state["stage"] + 1} first.\n'
                f'Clicked step {index + 1}. See terminal GUI CLICK log.')
            return

        # Never enter reverse reorientation unless we are physically at the
        # recorded endpoint, even if the GUI stage says otherwise.
        if index == 2:
            target = np.asarray(state['paths']['reorient'][-1])
            mismatch = float(np.max(np.abs(np.rad2deg(np.asarray(robot.q) - target))))
            if mismatch > 2.0:
                state['stage'] = 1
                status.set(f'Step 2 needs re-run: clearance mismatch {mismatch:.2f} deg')
                messagebox.showwarning('Not at clearance',
                    f'Robot is {mismatch:.2f} deg from saved clearance.\n'
                    'Step 2 has been unlocked. Run Step 2 again.')
                return
        try:
            steps[index][1]()
            state['stage'] += 1
            print(f'GUI SUCCESS | completed step {index+1} | next step {state["stage"]+1}', flush=True)
            if index > 0:
                status.set(f'Step {index + 1} complete. ' +
                           ('At CLOSED handle; awaiting release/standby.' if index == 5
                            else f'Next: step {index + 2}.'))
        except Exception as exc:
            traceback.print_exc()
            status.set(f'FAILED at step {index + 1}: {exc}')
            messagebox.showerror('Motion failed', str(exc))

    tk.Label(root, text='Door return | Reverse-path TEACH + closing',
             font=('Segoe UI', 16, 'bold')).pack(pady=(16, 5))
    tk.Label(root, text='Reverse recorded paths and synchronize door closing. No geometry collision checking.',
             fg='#ad6200').pack(pady=(0, 12))
    for i, (name, _) in enumerate(steps):
        tk.Button(root, text=name, command=lambda i=i: execute(i),
                  height=2, width=79).pack(pady=5)
    tk.Label(root, textvariable=status, fg='#086b94', wraplength=600).pack(pady=14)
    root.mainloop()


if __name__ == '__main__':
    main()
