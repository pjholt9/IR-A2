"""Reusable EC66 door-opening task (printer-relative station handled by Scenery)."""
from math import pi
import numpy as np
from spatialmath import SE3
from scipy import linalg
from scipy.spatial.transform import Rotation, Slerp
from roboticstoolbox import jtraj
from kinematic_analysis import KinematicRecorder


class RobotTasks:
    def __init__(self, scenery, env):
        self.scenery = scenery
        self.env = env
        self.robot = scenery.robot
        if self.robot is None:
            raise RuntimeError("Call scenery.add_to_env(env) before RobotTasks")

    def open_door(self, printer, move_to_station=True):
        """Open selected printer door; leave robot at handle and door open.

        Returns a KinematicRecorder for plots and viva analysis.
        Does not assert collision-free motion.
        """
        scenery = self.scenery
        env = self.env
        robot = self.robot
        door = printer.door
        if move_to_station:
            scenery.move_to_printer(printer, env)
        robot.q = scenery.ec66.q_standby.copy()
        door.update_angle(0)
        env.step(0.1)
        recorder = KinematicRecorder(robot)
        self._active_recorder = recorder
        self._active_printer = printer
        self._approach_rotation = (door.get_handle_pose() @ SE3.Rx(-pi / 2)).R.copy()
        self._retract_pose = None
        self._door_return_paths = {}
        self._door_opening_q = None
        self._door_approach_q = None
        self._door_pregrasp_q = None

        T_handle = door.get_handle_pose()



        T_contact = T_handle @ SE3.Rx(-pi/2)



        # 10 cm away from handle along EE local Z

        T_pregrasp = T_contact @ SE3(0, 0, -0.10)



        solution = robot.ikine_LM(

            T_pregrasp,

            q0=robot.q,

            joint_limits=True

        )



        print("Pre-grasp IK success:", solution.success)



        if not solution.success:

            raise RuntimeError("Pre-grasp IK failed")



        trajectory = jtraj(robot.q.copy(), solution.q, 200)



        for q in trajectory.q[1:]:

            robot.q = q

            env.step(0.02)



            recorder.record(0.02, "Pre-grasp")



        print("Reached pre-grasp")

        env.step(0.5)





        # -------------------------------------------------- #

        # Phase 2: Pre-grasp -> Handle (6-DoF Closed-loop RMRC)

        # -------------------------------------------------- #



        steps = 150

        delta_t = 0.03

        damping = 0.001

        Kp = 2.0

        KR = 2.0



        start_pos = robot.fkine(robot.q).t.copy()

        end_pos = T_contact.t.copy()



        positions = np.linspace(start_pos, end_pos, steps)



        for i in range(steps - 1):



            # Current EE pose

            T_actual = robot.fkine(robot.q)



            # ---------------------------------------------- #

            # Linear velocity (XYZ)

            # ---------------------------------------------- #



            x_desired = positions[i + 1]



            xdot_desired = (

                positions[i + 1] - positions[i]

            ) / delta_t



            position_error = x_desired - T_actual.t



            linear_velocity = (

                xdot_desired + Kp * position_error

            )



            # ---------------------------------------------- #

            # Angular velocity (orientation control)

            # ---------------------------------------------- #



            R_desired = T_contact.R

            R_actual = T_actual.R



            R_error = R_desired @ R_actual.T



            rotation_error = Rotation.from_matrix(

                R_error

            ).as_rotvec()



            angular_velocity = KR * rotation_error



            # ---------------------------------------------- #

            # Full 6-DoF Cartesian velocity

            # ---------------------------------------------- #



            xdot = np.r_[

                linear_velocity,

                angular_velocity

            ]



            J = robot.jacob0(robot.q)



            # Dimension checks

            assert J.shape == (6, 6)

            assert xdot.shape == (6,)



            # Damped Least Squares

            qdot = J.T @ linalg.solve(

                J @ J.T + damping * np.eye(6),

                xdot

            )



            q_next = robot.q + delta_t * qdot



            # ---------------------------------------------- #

            # Joint limit check

            # ---------------------------------------------- #



            within_limits = all(

                robot.links[j].qlim[0] <= q_next[j] <= robot.links[j].qlim[1]

                for j in range(robot.n)

            )



            if not within_limits:

                raise RuntimeError(

                    "RMRC stopped: joint limit reached"

                )



            robot.q = q_next



            env.step(delta_t)

            recorder.record(delta_t, "Approach")



        # -------------------------------------------------- #

        # Check final position and orientation errors

        # -------------------------------------------------- #



        T_actual = robot.fkine(robot.q)



        position_error = np.linalg.norm(

            T_contact.t - T_actual.t

        )



        orientation_error = np.linalg.norm(

            Rotation.from_matrix(

                T_contact.R @ T_actual.R.T

            ).as_rotvec()

        )



        print("\n========== END OF APPROACH ==========")

        print("Position error (mm):", position_error * 1000)

        print("Orientation error (deg):", np.rad2deg(orientation_error))

        print("====================================\n")



        if position_error > 0.005:

            raise RuntimeError(

                "EE did not reach handle accurately"

            )



        if orientation_error > np.deg2rad(2):

            raise RuntimeError(

                "EE orientation error exceeds 2 degrees"

            )



        print("Reached handle")

        env.step(0.5)

        # -------------------------------------------------- #

        # Phase 3: Open door (Continuous IK trajectory)

        # -------------------------------------------------- #



        angles = np.linspace(0, -pi/2, 91)



        q_solutions = []

        q_guess = robot.q.copy()



        # Calculate entire door-opening trajectory first

        for angle in angles:



            door.update_angle(angle)



            T_handle = door.get_handle_pose()

            T_target = T_handle @ SE3.Rx(-pi/2)



            solution = robot.ikine_LM(

                T_target,

                q0=q_guess,

                joint_limits=True

            )



            if not solution.success:

                door.update_angle(0)

                raise RuntimeError(

                    f"Door IK failed at {np.rad2deg(angle):.1f} degrees"

                )



            q_solutions.append(solution.q.copy())

            q_guess = solution.q.copy()



        # Reset door before animation

        door.update_angle(0)



        # Check joint continuity

        q_array = np.array(q_solutions)



        # Check continuity between RMRC approach and door IK

        joint_jump = np.rad2deg(

            q_array[0] - recorder.q[-1]

        )



        print(

            "Phase 2 -> Phase 3 joint jump (deg):",

            np.round(joint_jump, 3)

        )



        if np.max(np.abs(joint_jump)) > 2:

            raise RuntimeError(

                "Large joint jump between approach and door opening"

            )



        joint_changes = np.abs(np.diff(q_array, axis=0))

        max_change = np.rad2deg(np.max(joint_changes))



        print("Maximum joint change per step:", max_change, "deg")



        if max_change > 10:

            raise RuntimeError("Large joint jump detected in door trajectory")







        # Animate door and robot together

        for i in range(1, len(angles)):



            door.update_angle(angles[i])

            robot.q = q_solutions[i].copy()



            env.step(0.05)



            recorder.record(0.05, "Door opening")



        # Preserve BOTH original approach segments, including their starting joint samples.
        # Pre-grasp: standby -> pre-grasp (jtraj).
        # Approach: pre-grasp -> closed-door handle (DLS RMRC).
        pre_ids = [i for i, name in enumerate(recorder.phase) if name == "Pre-grasp"]
        approach_ids = [i for i, name in enumerate(recorder.phase) if name == "Approach"]
        if not pre_ids or not approach_ids or pre_ids[0] < 1 or approach_ids[0] < 1:
            raise RuntimeError("Cannot save door approach: missing pre-grasp/approach samples")
        if approach_ids[0] != pre_ids[-1] + 1:
            raise RuntimeError("Door approach phase samples are not contiguous")
        self._door_pregrasp_q = np.asarray(
            [recorder.q[pre_ids[0] - 1]] + [recorder.q[i] for i in pre_ids], dtype=float).copy()
        self._door_approach_q = np.asarray(
            [recorder.q[approach_ids[0] - 1]] + [recorder.q[i] for i in approach_ids], dtype=float).copy()
        print(f"SAVED DOOR APPROACH | pre-grasp {len(self._door_pregrasp_q)} samples | "
              f"RMRC {len(self._door_approach_q)} samples")

        # Snapshot verified opening path BEFORE other task phases append to recorder.
        opening_ids = [i for i, name in enumerate(recorder.phase) if name == "Door opening"]
        if len(opening_ids) == 90 and opening_ids[0] > 0:
            self._door_opening_q = np.asarray(
                [recorder.q[opening_ids[0] - 1]] + [recorder.q[i] for i in opening_ids], dtype=float).copy()
        else:
            raise RuntimeError(f"Cannot save door opening trajectory: {len(opening_ids)} samples")
        print("Door fully opened!")

        env.step(1)
        return recorder


    # Clearance displacement expressed in the PRINTER frame, not world frame.
    # For the current Rz(pi) printers this equals world [-0.20, +0.30, +0.20] m.
    CLEARANCE_OFFSET_PRINTER = np.array([0.20, -0.30, 0.20])

    def _task_recorder(self, printer, recorder=None):
        if getattr(self, "_active_printer", None) is not printer:
            raise RuntimeError("Call open_door(printer) before the return sequence")
        if recorder is None:
            recorder = getattr(self, "_active_recorder", None)
        if recorder is None:
            raise RuntimeError("No active kinematic recorder")
        return recorder

    def _move_jtraj(self, q_goal, recorder, label, steps=200, dt=0.02):
        robot = self.robot
        qlim = np.asarray([link.qlim for link in robot.links], dtype=float)
        trajectory = jtraj(robot.q.copy(), np.asarray(q_goal, dtype=float), steps).q
        # Validate the entire joint-space path before commanding movement.
        if np.any(trajectory < qlim[:, 0]) or np.any(trajectory > qlim[:, 1]):
            raise RuntimeError(f"Joint limit violation during {label}")
        for q in trajectory[1:]:
            robot.q = np.asarray(q, dtype=float).copy()
            self.env.step(dt)
            recorder.record(dt, label)

    def retract_from_door(self, printer, distance=0.10, recorder=None):
        """RMRC retreat along the open handle EE's negative local Z axis.

        Requires open_door(printer) to have finished. Returns target SE3.
        """
        recorder = self._task_recorder(printer, recorder)
        _saved_start_q = self.robot.q.copy()
        _saved_rec_index = len(recorder.q)
        robot, door = self.robot, printer.door
        T_handle_open = door.get_handle_pose() @ SE3.Rx(-pi / 2)
        T_goal = T_handle_open @ SE3(0, 0, -float(distance))
        T_start = robot.fkine(robot.q)
        pos_error = np.linalg.norm(T_start.t - T_handle_open.t)
        print(f"Open-handle starting position error: {pos_error*1000:.2f} mm")
        if pos_error > 0.005:
            raise RuntimeError("EE is not at open handle; aborting retract")

        steps, dt, damping, kp, kr = 150, 0.03, 0.001, 2.0, 2.0
        positions = np.linspace(T_start.t, T_goal.t, steps)
        qlim = np.asarray([link.qlim for link in robot.links], dtype=float)
        for i in range(steps - 1):
            actual = robot.fkine(robot.q)
            v = ((positions[i+1] - positions[i]) / dt
                 + kp * (positions[i+1] - actual.t))
            w = kr * Rotation.from_matrix(T_goal.R @ actual.R.T).as_rotvec()
            J = robot.jacob0(robot.q)
            qdot = J.T @ linalg.solve(J @ J.T + damping * np.eye(6), np.r_[v, w])
            q_next = robot.q + dt * qdot
            if np.any(q_next < qlim[:, 0]) or np.any(q_next > qlim[:, 1]):
                raise RuntimeError(f"Retract RMRC joint limit at step {i}")
            robot.q = q_next
            self.env.step(dt)
            recorder.record(dt, "Retract")

        T_final = robot.fkine(robot.q)
        pos = np.linalg.norm(T_goal.t - T_final.t)
        rot = np.linalg.norm(Rotation.from_matrix(T_goal.R @ T_final.R.T).as_rotvec())
        print(f"Retract target error: {pos*1000:.2f} mm, {np.rad2deg(rot):.3f} deg")
        if pos > 0.005 or rot > np.deg2rad(2):
            raise RuntimeError("Retract tracking tolerance exceeded")
        self._retract_pose = T_goal
        self._door_return_paths["retract"] = np.asarray(
            [_saved_start_q] + [np.asarray(q, dtype=float).copy() for q in recorder.q[_saved_rec_index:]],
            dtype=float)
        self._door_return_paths["retract"][-1] = self.robot.q.copy()
        return T_goal

    def move_to_clearance(self, printer, offset=None, recorder=None):
        """IK + jtraj to a printer-frame displacement from retract target.

        offset: [dx,dy,dz] in printer frame, metres. Defaults to taught pose.
        """
        recorder = self._task_recorder(printer, recorder)
        _saved_start_q = self.robot.q.copy()
        _saved_rec_index = len(recorder.q)
        if self._retract_pose is None:
            raise RuntimeError("Call retract_from_door before move_to_clearance")
        offset = (self.CLEARANCE_OFFSET_PRINTER if offset is None
                  else np.asarray(offset, dtype=float))
        offset = np.asarray(offset, dtype=float).reshape(3)
        # Convert only the displacement; target is relative to the retract EE position.
        world_displacement = printer.pose.R @ offset
        T_goal = SE3.Rt(self._retract_pose.R,
                        np.asarray(self._retract_pose.t) + world_displacement)
        sol = self.robot.ikine_LM(T_goal, q0=self.robot.q.copy(), joint_limits=True)
        if not sol.success:
            raise RuntimeError("Clearance waypoint IK failed; no motion executed")
        self._move_jtraj(sol.q, recorder, "Clearance waypoint")
        err = np.linalg.norm(self.robot.fkine(self.robot.q).t - T_goal.t) * 1000
        print(f"Clearance offset (printer frame): {offset} m")
        print(f"Clearance EE position error: {err:.2f} mm")
        self._door_return_paths["clearance"] = np.asarray(
            [_saved_start_q] + [np.asarray(q, dtype=float).copy() for q in recorder.q[_saved_rec_index:]],
            dtype=float)
        self._door_return_paths["clearance"][-1] = self.robot.q.copy()
        return T_goal

    def reorient_for_standby(self, printer, recorder=None, steps=60):
        """Keep EE XYZ fixed while SLERP-ing back to the closed-door approach rotation."""
        recorder = self._task_recorder(printer, recorder)
        _saved_start_q = self.robot.q.copy()
        _saved_rec_index = len(recorder.q)
        robot = self.robot
        R_target = self._approach_rotation
        T_start = robot.fkine(robot.q)
        xyz = np.asarray(T_start.t, dtype=float).copy()
        key_rots = Rotation.from_matrix(np.stack((T_start.R, R_target)))
        slerp = Slerp([0.0, 1.0], key_rots)
        q_prev = robot.q.copy()
        q_path = []
        qlim = np.asarray([link.qlim for link in robot.links], dtype=float)
        for fraction in np.linspace(0.0, 1.0, steps)[1:]:
            R = slerp([fraction]).as_matrix()[0]
            T_goal = SE3.Rt(R, xyz)
            sol = robot.ikine_LM(T_goal, q0=q_prev, joint_limits=True)
            if not sol.success:
                raise RuntimeError(f"Reorientation IK failed at {fraction:.1%}; no rotation executed")
            q_next = np.asarray(sol.q, dtype=float)
            if np.any(q_next < qlim[:, 0]) or np.any(q_next > qlim[:, 1]):
                raise RuntimeError("Reorientation joint limit violation; no rotation executed")
            step_deg = np.rad2deg(np.max(np.abs(q_next - q_prev)))
            if step_deg > 8.0:
                raise RuntimeError(f"Reorientation IK jump {step_deg:.2f} deg; no rotation executed")
            q_path.append(q_next)
            q_prev = q_next
        for q in q_path:
            robot.q = q.copy()
            self.env.step(0.03)
            recorder.record(0.03, "Reorient EE")
        T_end = robot.fkine(robot.q)
        pos_error = np.linalg.norm(T_end.t - xyz) * 1000
        angle_error = np.rad2deg(np.linalg.norm(
            Rotation.from_matrix(R_target @ T_end.R.T).as_rotvec()))
        print(f"Reorient: XYZ drift {pos_error:.2f} mm, orientation error {angle_error:.3f} deg")
        self._door_return_paths["reorient"] = np.asarray(
            [_saved_start_q] + [np.asarray(q, dtype=float).copy() for q in recorder.q[_saved_rec_index:]],
            dtype=float)
        self._door_return_paths["reorient"][-1] = self.robot.q.copy()
        return T_end

    def return_to_standby(self, printer, recorder=None):
        """Joint-space return to standby; printer door remains open."""
        recorder = self._task_recorder(printer, recorder)
        self._move_jtraj(self.scenery.ec66.q_standby.copy(), recorder, "Return standby")
        print("Standby reached; printer door remains open")
        return self.robot.q.copy()

    def _reverse_saved_door_segment(self, printer, label, recorder=None, dt=0.03):
        """Replay a captured joint-space segment backwards (no fresh IK)."""
        recorder = self._task_recorder(printer, recorder)
        if label not in self._door_return_paths:
            raise RuntimeError(f"No saved {label} trajectory; run opening tasks first")
        path = self._door_return_paths[label]
        mismatch = float(np.rad2deg(np.max(np.abs(self.robot.q - path[-1]))))
        print(f"Reverse {label} | start mismatch {mismatch:.6f} deg")
        if mismatch > 2.0:
            raise RuntimeError(f"Reverse {label} aborted: start mismatch {mismatch:.3f} deg")
        for q in path[-2::-1]:
            self.robot.q = q.copy()
            self.env.step(dt)
            recorder.record(dt, f"Reverse {label}")
        return self.robot.fkine(self.robot.q)

    def return_to_open_handle(self, printer, recorder=None, steps=220):
        """Standby -> saved reorientation endpoint -> reverse 3 verified paths."""
        recorder = self._task_recorder(printer, recorder)
        if any(k not in self._door_return_paths for k in ('reorient', 'clearance', 'retract')):
            raise RuntimeError('Missing saved return segments')
        q_target = self._door_return_paths['reorient'][-1]
        q_start = np.asarray(self.robot.q, dtype=float).copy()
        path = np.asarray(jtraj(q_start, q_target, steps).q, dtype=float)
        qlim = np.asarray(self.robot.qlim, dtype=float)
        if qlim.shape != (2, len(q_start)):
            raise RuntimeError(f'Unexpected joint limits shape {qlim.shape}')
        low, high = qlim
        if np.any((np.isfinite(low) & (path < low - 1e-6)) |
                  (np.isfinite(high) & (path > high + 1e-6))):
            raise RuntimeError('Standby -> saved clearance joint limit violation')
        for q in path[1:]:
            self.robot.q = q.copy()
            self.env.step(0.02)
            recorder.record(0.02, 'Standby to saved clearance')
        for name in ('reorient', 'clearance', 'retract'):
            self._reverse_saved_door_segment(printer, name, recorder)
        actual = self.robot.fkine(self.robot.q)
        target = printer.door.get_handle_pose() @ SE3.Rx(-pi/2)
        p_err = float(np.linalg.norm(actual.t - target.t) * 1000)
        r_err = float(np.rad2deg(np.linalg.norm(
            Rotation.from_matrix(target.R @ actual.R.T).as_rotvec())))
        print(f'OPEN HANDLE REACHED | XYZ error {p_err:.3f} mm | R error {r_err:.4f} deg')
        if p_err > 5.0 or r_err > 2.0:
            raise RuntimeError('Return-to-handle alignment failed; closing prohibited')
        return actual

    def close_door(self, printer, recorder=None):
        """Replay verified opening trajectory backwards, synchronized with door angle."""
        recorder = self._task_recorder(printer, recorder)
        opening_q = self._door_opening_q
        if opening_q is None or opening_q.shape != (91, len(self.robot.q)):
            raise RuntimeError('Missing saved 91-point door opening path')
        mismatch = float(np.rad2deg(np.max(np.abs(self.robot.q - opening_q[-1]))))
        print(f'CLOSE START | joint mismatch {mismatch:.6f} deg')
        if mismatch > 2.0:
            raise RuntimeError('Door closing aborted: robot not at recorded open handle')
        angles = np.linspace(0.0, -pi/2, 91)
        worst_mm = 0.0
        worst_deg = 0.0
        for i in range(89, -1, -1):
            printer.door.update_angle(float(angles[i]))
            self.robot.q = opening_q[i].copy()
            self.env.step(0.05)
            recorder.record(0.05, 'Door closing (reverse playback)')
            target = printer.door.get_handle_pose() @ SE3.Rx(-pi/2)
            actual = self.robot.fkine(self.robot.q)
            err_mm = float(np.linalg.norm(actual.t - target.t) * 1000)
            err_deg = float(np.rad2deg(np.linalg.norm(
                Rotation.from_matrix(target.R @ actual.R.T).as_rotvec())))
            worst_mm = max(worst_mm, err_mm)
            worst_deg = max(worst_deg, err_deg)
            if i % 15 == 0:
                print(f'CLOSING | angle={np.rad2deg(angles[i]):.1f} deg | '
                      f'handle error={err_mm:.3f} mm, {err_deg:.4f} deg')
            if err_mm > 5.0 or err_deg > 2.0:
                raise RuntimeError('Door closing tracking error exceeded tolerance')
        print(f'DOOR CLOSED | worst handle error {worst_mm:.3f} mm / {worst_deg:.4f} deg')
        return True

    def release_closed_handle_and_standby(self, printer, recorder=None, dt=0.03,
                                          tolerance_deg=2.0):
        """Replay closed-handle approach and pre-grasp backwards, ending at standby.

        Requires close_door() to have completed. No new IK or Jacobian inversion.
        Keeps the printer door closed and does not move the linear platform.
        """
        recorder = self._task_recorder(printer, recorder)
        if self._door_approach_q is None or self._door_pregrasp_q is None:
            raise RuntimeError("No saved door approach; run open_door first")

        for label, path in (("Reverse approach (closed handle to pre-grasp)",
                             self._door_approach_q),
                            ("Reverse pre-grasp (to standby)",
                             self._door_pregrasp_q)):
            mismatch = float(np.rad2deg(np.max(np.abs(self.robot.q - path[-1]))))
            print(f"{label} | start joint mismatch {mismatch:.6f} deg")
            if mismatch > tolerance_deg:
                raise RuntimeError(f"{label}: start mismatch {mismatch:.3f} deg")
            for q in path[-2::-1]:
                self.robot.q = np.asarray(q, dtype=float).copy()
                self.env.step(dt)
                recorder.record(dt, label)

        standby = np.asarray(self.scenery.ec66.q_standby, dtype=float)
        err = float(np.rad2deg(np.max(np.abs(self.robot.q - standby))))
        print(f"CLOSED DOOR RELEASE COMPLETE | standby error {err:.6f} deg")
        if err > 0.1:
            raise RuntimeError(f"Reverse pre-grasp did not reach standby ({err:.3f} deg)")
        return self.robot.q.copy()

    def finish_cycle_and_home(self, printer, recorder=None):
        """Return from closed handle to standby, then rail home (robot stays standby)."""
        self.release_closed_handle_and_standby(printer, recorder=recorder)
        self.scenery.home(self.env)
        print("FULL CYCLE COMPLETE | door closed | robot standby | rail home")
        return True

    # Camera pose is defined in the calibrated plate-reference frame.
    CAMERA_XYZ_PLATE = np.array([-0.125, -0.150, 0.225])
    CAMERA_RPY_DEG = np.array([-120.0, 0.0, 0.0])
    CAMERA_WAYPOINT_OFFSET_WORLD = np.array([-0.15, 0.0, 0.15])

    def _camera_target(self, printer):
        r, p, y = np.deg2rad(self.CAMERA_RPY_DEG)
        return (printer.plate_reference @ SE3(*self.CAMERA_XYZ_PLATE)
                @ SE3.Rx(r) @ SE3.Ry(p) @ SE3.Rz(y))

    def _camera_cartesian_rmrc(self, T_goal, recorder, label, steps=180, dt=0.03,
                               damping=0.001, kp=2.0, kr=2.0):
        """Closed-loop 6D Cartesian tracking with linear XYZ and SLERP orientation.

        This is a kinematic simulation, not collision detection or a safety controller.
        """
        robot = self.robot
        T_start = robot.fkine(robot.q)
        positions = np.linspace(T_start.t, T_goal.t, steps)
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(
            np.stack((T_start.R, T_goal.R))))
        qlim = np.asarray([link.qlim for link in robot.links], dtype=float)
        for i in range(steps - 1):
            actual = robot.fkine(robot.q)
            fraction = (i + 1) / (steps - 1)
            R_des = slerp([fraction]).as_matrix()[0]
            v = ((positions[i + 1] - positions[i]) / dt
                 + kp * (positions[i + 1] - actual.t))
            w = kr * Rotation.from_matrix(R_des @ actual.R.T).as_rotvec()
            J = robot.jacob0(robot.q)
            qdot = J.T @ linalg.solve(
                J @ J.T + damping * np.eye(6), np.r_[v, w])
            q_next = robot.q + dt * qdot
            if not np.all(np.isfinite(q_next)):
                raise RuntimeError(f"{label}: non-finite RMRC solution")
            if np.any(q_next < qlim[:, 0]) or np.any(q_next > qlim[:, 1]):
                raise RuntimeError(f"{label}: joint limit reached at step {i}")
            robot.q = q_next
            self.env.step(dt)
            recorder.record(dt, label)
        T_end = robot.fkine(robot.q)
        pos_mm = 1000 * np.linalg.norm(T_goal.t - T_end.t)
        rot_deg = np.rad2deg(np.linalg.norm(
            Rotation.from_matrix(T_goal.R @ T_end.R.T).as_rotvec()))
        print(f"{label}: position error {pos_mm:.2f} mm, orientation error {rot_deg:.3f} deg")
        if pos_mm > 5.0 or rot_deg > 2.0:
            raise RuntimeError(f"{label}: tracking tolerance exceeded")
        return T_end

    def move_to_camera_pose(self, printer, recorder=None):
        """Standby -> camera waypoint (jtraj) -> camera EE pose (6D RMRC).

        Call after return_to_standby(printer). Door must remain open.
        Returns the actual EE pose at the camera target.
        """
        recorder = self._task_recorder(printer, recorder)
        if np.max(np.abs(self.robot.q - self.scenery.ec66.q_standby)) > np.deg2rad(2):
            raise RuntimeError("Camera task must start from standby")

        T_camera = self._camera_target(printer)
        T_waypoint = SE3.Rt(
            T_camera.R,
            T_camera.t + self.CAMERA_WAYPOINT_OFFSET_WORLD)
        sol_waypoint = self.robot.ikine_LM(
            T_waypoint, q0=self.robot.q.copy(), joint_limits=True)
        if not sol_waypoint.success:
            raise RuntimeError("Camera waypoint IK failed")
        sol_camera = self.robot.ikine_LM(
            T_camera, q0=sol_waypoint.q.copy(), joint_limits=True)
        if not sol_camera.success:
            raise RuntimeError("Camera target IK failed")
        print("Camera waypoint and target IK: SUCCESS")
        self._move_jtraj(sol_waypoint.q, recorder, "Camera waypoint", steps=200, dt=0.02)
        T_actual = self._camera_cartesian_rmrc(
            T_camera, recorder, "Camera RMRC", steps=180, dt=0.03)
        self._camera_pose = T_camera
        self._pre_pick_pose = None
        return T_actual

    def camera_capture_dwell(self, printer, seconds=3.0, recorder=None):
        """Hold camera EE pose while simulating image capture (no camera API)."""
        recorder = self._task_recorder(printer, recorder)
        if getattr(self, "_camera_pose", None) is None:
            raise RuntimeError("Call move_to_camera_pose first")
        if seconds < 0:
            raise ValueError("Dwell must be nonnegative")
        remaining = float(seconds)
        while remaining > 1e-9:
            dt = min(0.03, remaining)
            self.env.step(dt)
            remaining -= dt
        print(f"Camera capture dwell complete: {seconds:.2f} s")
        return self.robot.fkine(self.robot.q)

    def move_to_pre_pick(self, printer, distance=0.10, recorder=None):
        """Retract EE in WORLD +Y while preserving camera orientation.

        Returns the actual world-frame pre-pick EE pose. It is NOT a part pose.
        """
        recorder = self._task_recorder(printer, recorder)
        if getattr(self, "_camera_pose", None) is None:
            raise RuntimeError("Call move_to_camera_pose first")
        if distance <= 0:
            raise ValueError("Pre-pick retreat distance must be positive")
        T_start = self.robot.fkine(self.robot.q)
        if np.linalg.norm(T_start.t - self._camera_pose.t) > 0.005:
            raise RuntimeError("EE is not at camera pose; cannot start pre-pick retreat")
        T_goal = SE3(0.0, float(distance), 0.0) @ T_start
        sol = self.robot.ikine_LM(
            T_goal, q0=self.robot.q.copy(), joint_limits=True)
        if not sol.success:
            raise RuntimeError("Pre-pick pose IK failed; no motion executed")
        T_actual = self._camera_cartesian_rmrc(
            T_goal, recorder, "Pre-pick retreat", steps=120, dt=0.03)
        self._pre_pick_pose = T_actual
        print("Pre-pick EE world XYZ (m):", np.round(T_actual.t, 5))
        return T_actual


    # ----- PLAN B: deterministic part pick / straight RMRC -----
    PICK_APPROACH_M = 0.250
    PICK_RETREAT_M = 0.600
    PICK_RPY_DEG = (-90.0, 0.0, 0.0)  # relative to printer.plate_reference

    def pick_target_pose(self, printer, part):
        """Pick position from STL-local geometry, orientation from printer frame."""
        from printed_part import pick_local_pose
        T_part = SE3(np.asarray(part.mesh.T, dtype=float))
        xyz = (T_part @ pick_local_pose(part.kind)).t
        rx, ry, rz = np.deg2rad(self.PICK_RPY_DEG)
        R = (printer.plate_reference.R @ SE3.Rx(rx).R
             @ SE3.Ry(ry).R @ SE3.Rz(rz).R)
        return SE3.Rt(R, xyz)

    def _plan_linear_pick_rmrc(self, q_start, T_goal, steps=220, dt=0.03,
                               damping=0.002, kp=2.0, kr=2.0):
        """OFFLINE DLS RMRC with straight Cartesian XYZ and fixed orientation."""
        q = np.asarray(q_start, dtype=float).copy()
        T_start = self.robot.fkine(q)
        R_fixed = T_start.R.copy()
        rot_diff = Rotation.from_matrix(T_goal.R @ R_fixed.T).as_rotvec()
        if np.linalg.norm(rot_diff) > np.deg2rad(2):
            raise RuntimeError("RMRC segment orientation mismatch")
        xyz = np.linspace(T_start.t, T_goal.t, steps)
        qlim = np.asarray([link.qlim for link in self.robot.links], dtype=float)
        path = [q.copy()]
        for i in range(steps - 1):
            T_now = self.robot.fkine(q)
            v = ((xyz[i + 1] - xyz[i]) / dt
                 + kp * (xyz[i + 1] - T_now.t))
            w = kr * Rotation.from_matrix(R_fixed @ T_now.R.T).as_rotvec()
            J = self.robot.jacob0(q)
            qdot = J.T @ linalg.solve(J @ J.T + damping * np.eye(6),
                                      np.r_[v, w])
            q_next = q + dt * qdot
            if not np.all(np.isfinite(q_next)):
                raise RuntimeError(f"RMRC nonfinite joint solution at {i}")
            if np.any(q_next < qlim[:, 0]) or np.any(q_next > qlim[:, 1]):
                raise RuntimeError(f"RMRC joint limit at {i}")
            if np.max(np.abs(q_next - q)) > np.deg2rad(7):
                raise RuntimeError(f"RMRC joint jump > 7 degrees at {i}")
            q = q_next
            path.append(q.copy())
        T_end = self.robot.fkine(q)
        p_err = np.linalg.norm(T_goal.t - T_end.t)
        r_err = np.linalg.norm(Rotation.from_matrix(
            T_goal.R @ T_end.R.T).as_rotvec())
        if p_err > 0.005 or r_err > np.deg2rad(2):
            raise RuntimeError(f"RMRC error {p_err*1000:.2f}mm / "
                               f"{np.rad2deg(r_err):.2f}deg")
        return np.asarray(path)

    def plan_part_pick(self, printer, part, approach_m=None, retreat_m=None):
        """Preview ALL segments before moving; requires pre-pick pose."""
        if getattr(self, "_pre_pick_pose", None) is None:
            raise RuntimeError("Run move_to_pre_pick before planning")
        a = self.PICK_APPROACH_M if approach_m is None else float(approach_m)
        r = self.PICK_RETREAT_M if retreat_m is None else float(retreat_m)
        if a <= 0 or r < a:
            raise ValueError("Require 0 < approach <= retreat")
        T_pick = self.pick_target_pose(printer, part)
        T_out = SE3.Rt(T_pick.R, T_pick.t + np.array([0, a, 0]))
        T_retreat = SE3.Rt(T_pick.R, T_pick.t + np.array([0, r, 0]))
        sol = self.robot.ikine_LM(T_out, q0=self.robot.q.copy(), joint_limits=True)
        if not sol.success:
            raise RuntimeError("Outside alignment IK failed")
        q_out = np.asarray(sol.q, dtype=float)
        align = np.asarray(jtraj(self.robot.q.copy(), q_out, 180).q)
        inward = self._plan_linear_pick_rmrc(q_out, T_pick, steps=220)
        outward = self._plan_linear_pick_rmrc(inward[-1], T_retreat, steps=340)
        qlim = np.asarray([link.qlim for link in self.robot.links], dtype=float)
        for label, path in (("align", align), ("in", inward),
                            ("out", outward)):
            if np.any(path < qlim[:, 0]) or np.any(path > qlim[:, 1]):
                raise RuntimeError(f"{label}: joint limit violation")
        return {"align": align, "in": inward, "out": outward,
                "target": T_pick}

    def execute_part_pick(self, printer, part, plan=None, recorder=None):
        """Approach, verify grasp (<1 mm), attach, retreat and HOLD.

        No standby return: drop waypoint will be taught separately.
        On failed grasp, stop at pick point; do not retreat as if carrying.
        """
        recorder = self._task_recorder(printer, recorder)
        if plan is None:
            plan = self.plan_part_pick(printer, part)
        if np.max(np.abs(self.robot.q - plan["align"][0])) > 1e-5:
            raise RuntimeError("Robot moved since planning; replan required")
        if part.isAttached:
            raise RuntimeError("Part is already attached")
        for phase in ("align", "in", "out"):
            if phase == "out" and not part.isAttached:
                raise RuntimeError("ATTACH FAILED: retreat cancelled; part not attached")
            for q in plan[phase][1:]:
                self.robot.q = np.asarray(q, dtype=float).copy()
                if part.isAttached:
                    part.follow_ee()
                self.env.step(0.03)
                recorder.record(0.03, "Pick " + phase)
            if phase == "in":
                if not part.attach_to(self.robot, grasp_world_pose=plan["target"],
                                      tolerance_m=0.001):
                    print("ATTACH FAILED: EE-to-grasp error >= 1 mm. "
                          "Robot stopped at grasp pose; no retreat executed.")
                    return False
        self._post_pick_hold_pose = self.robot.fkine(self.robot.q)
        print("PICK COMPLETE: retreat finished; holding pose with part attached. "
              "Door remains open. Await pre-drop teaching.")
        print("Hold EE world XYZ (m):", np.round(self._post_pick_hold_pose.t, 5))
        return True

    # ---------------------------------------------------------------
    # Geometry-based pre-drop and vertical RMRC (validated in teach GUI)
    # ---------------------------------------------------------------
    DROP_CLEARANCE_M = {"barrel": 0.050, "body": 0.080}

    def plan_part_drop(self, part, bucket_pose, floor_z, bucket_width=0.24,
                       wall_thickness=0.02, clearance_m=None,
                       pre_drop_pose=None):
        """Preview pre-drop jtraj and vertical DLS RMRC; NO collision checking.

        Must start at the validated post-pick hold pose, with part attached.
        Floor Z is the INSIDE floor TOP, not PH resting-centre Z.
        """
        from printed_part import PART_CONFIG
        if not part.isAttached:
            raise RuntimeError("DROP PLAN FAILED: part is not attached")
        if part.kind not in self.DROP_CLEARANCE_M:
            raise ValueError(f"Unknown part kind: {part.kind}")
        if pre_drop_pose is None:
            raise ValueError("Pass the taught world-frame pre_drop_pose")
        clearance = (self.DROP_CLEARANCE_M[part.kind] if clearance_m is None
                     else float(clearance_m))
        if not np.isfinite(clearance) or not 0 <= clearance <= 0.12:
            raise ValueError("Release clearance must be between 0 and 120 mm")
        T_pre = SE3(pre_drop_pose.A)
        q_start = np.asarray(self.robot.q, dtype=float).copy()
        sol = self.robot.ikine_LM(T_pre, q0=q_start, joint_limits=True)
        if not sol.success:
            raise RuntimeError("Pre-drop IK failed")
        q_pre = np.asarray(sol.q, dtype=float)
        q_align = np.asarray(jtraj(q_start, q_pre, 240).q)
        singular_values = np.array([
            np.linalg.svd(self.robot.jacob0(q), compute_uv=False)
            for q in q_align
        ])

        sigma_min = singular_values[:, -1]

        print(
            "Pre-drop minimum Jacobian singular value:",
            np.min(sigma_min)
        )

        print(
            "Pre-drop worst sample:",
            np.argmin(sigma_min),
            "/",
            len(q_align) - 1
        )
        # Predict carried mesh pose at pre-drop WITHOUT changing the simulation.
        T_ee_now = self.robot.fkine(q_start)
        T_mesh_now = SE3(np.asarray(part.mesh.T, dtype=float))
        T_ee_mesh = T_ee_now.inv() @ T_mesh_now
        T_mesh_pre = T_pre @ T_ee_mesh
        cfg = PART_CONFIG[part.kind]
        w, d = float(cfg['width']), float(cfg['depth'])
        corners = np.array([[0., 0., 0., 1.], [w, 0., 0., 1.],
                            [0., d, 0., 1.], [w, d, 0., 1.]]).T
        world_corners = (T_mesh_pre.A @ corners)[:3, :]
        bucket_xy = np.asarray(bucket_pose.t[:2], dtype=float)
        half_opening = bucket_width / 2.0 - wall_thickness
        if np.any(np.abs(world_corners[:2, :] - bucket_xy[:, None]) > half_opening):
            raise RuntimeError("Part footprint outside bucket inner opening")
        bottom_z = float(np.min(world_corners[2, :]))
        travel = bottom_z - float(floor_z) - clearance
        if not np.isfinite(travel) or not 0 < travel <= 0.40:
            raise RuntimeError(f"Invalid drop travel: {travel*1000:.1f} mm")
        T_down = SE3.Rt(T_pre.R.copy(), T_pre.t + np.array([0., 0., -travel]))
        q_down = self._plan_linear_pick_rmrc(
            q_pre, T_down, steps=max(120, int(np.ceil(travel / 0.0015))))
        # Reuse the exact joint samples in reverse for the upward retreat.
        q_up = q_down[::-1].copy()
        print(f"DROP PLAN | {part.kind} | clearance {clearance*1000:.1f} mm "
              f"| travel {travel*1000:.1f} mm | floor Z {floor_z:.4f} m")
        return dict(align=q_align, down=q_down, up=q_up,
                    q_start=q_start, T_pre=T_pre, T_down=T_down,
                    distance=travel, floor_z=float(floor_z),
                    clearance=clearance, bucket_pose=SE3(bucket_pose.A))

    def execute_part_drop(self, part, plan, recorder=None):
        """Pre-drop jtraj, RMRC down, detach/settle, reverse RMRC up; HOLD.

        Returns True on completion. No standby, door closing or rail homing.
        """
        if not part.isAttached:
            raise RuntimeError("DROP ABORTED: part not attached")
        if np.max(np.abs(np.asarray(self.robot.q) - plan['q_start'])) > 1e-5:
            raise RuntimeError("Robot moved since drop planning; replan")
        recorder = recorder if recorder is not None else getattr(self, '_active_recorder', None)
        def step(q, label):
            self.robot.q = np.asarray(q, dtype=float).copy()
            if part.isAttached:
                part.follow_ee()
            self.env.step(0.03)
            if recorder is not None:
                recorder.record(0.03, label)
        for q in plan['align'][1:]:
            step(q, 'Pre-drop alignment')
        for q in plan['down'][1:]:
            step(q, 'Drop RMRC down')
        # Use the ACTUAL part mesh transform after motion, not predicted geometry.
        from printed_part import PART_CONFIG
        cfg = PART_CONFIG[part.kind]
        w, d = float(cfg['width']), float(cfg['depth'])
        corners = np.array([[0., 0., 0., 1.], [w, 0., 0., 1.],
                            [0., d, 0., 1.], [w, d, 0., 1.]]).T
        T_mesh = np.asarray(part.mesh.T, dtype=float).copy()
        bottom_z = float(np.min((T_mesh @ corners)[2, :]))
        settle = bottom_z - plan['floor_z']
        if settle < -0.005:
            raise RuntimeError("DROP ABORTED: part bottom penetrated bucket floor")
        part.detach()
        T_mesh[:3, 3] -= np.array([0., 0., settle])
        part.mesh.T = T_mesh
        print(f"DETACH SUCCESS | {part.kind} | settle {settle*1000:.1f} mm "
              f"| isAttached={part.isAttached}")
        for q in plan['up'][1:]:
            step(q, 'Drop RMRC up')
        print("DROP COMPLETE: EE at pre-drop, part in bucket; no standby yet")
        return True
