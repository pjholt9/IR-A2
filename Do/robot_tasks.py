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
        return T_goal

    def move_to_clearance(self, printer, offset=None, recorder=None):
        """IK + jtraj to a printer-frame displacement from retract target.

        offset: [dx,dy,dz] in printer frame, metres. Defaults to taught pose.
        """
        recorder = self._task_recorder(printer, recorder)
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
        return T_goal

    def reorient_for_standby(self, printer, recorder=None, steps=60):
        """Keep EE XYZ fixed while SLERP-ing back to the closed-door approach rotation."""
        recorder = self._task_recorder(printer, recorder)
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
        return T_end

    def return_to_standby(self, printer, recorder=None):
        """Joint-space return to standby; printer door remains open."""
        recorder = self._task_recorder(printer, recorder)
        self._move_jtraj(self.scenery.ec66.q_standby.copy(), recorder, "Return standby")
        print("Standby reached; printer door remains open")
        return self.robot.q.copy()
