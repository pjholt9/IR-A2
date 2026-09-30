import numpy as np

from math import pi
from spatialmath import SE3
from roboticstoolbox import DHRobot, DHLink, jtraj
from ir_support import CylindricalDHRobotPlot


class TX2_60:
    COLOURS = ("red", "green", "blue")
    APPROACH_HEIGHT = 0.15

    def __init__(self, base=SE3()):
        self.robot = None
        self.tx2_60_plot = None
        self.base = base
        self.q_standby = np.zeros(6)
        # Object currently held by the gripper, and the grasp offset relative to the end-effector
        self.attached_object = None
        self.grasp_offset = SE3()

    def update_base(self):
        """Sync the robot's base transform with self.base."""
        self.robot.base = self.base
        return self.robot.base

    def create_robot(self):
        """Build the DH model of the Staubli TX2-60 arm."""
        l1 = DHLink(d=0.375, a=0.050, alpha=-pi / 2, qlim=[-pi, pi])
        l2 = DHLink(d=0.0, a=0.290, alpha=0, qlim=[-127.5 * pi / 180, 127.5 * pi / 180])
        l3 = DHLink(d=0.0, a=0.020, alpha=-pi / 2, qlim=[-142.5 * pi / 180, 142.5 * pi / 180])
        l4 = DHLink(d=0.310, a=0.0, alpha=pi / 2, qlim=[-270 * pi / 180, 270 * pi / 180])
        l5 = DHLink(d=0.0, a=0.0, alpha=-pi / 2, qlim=[-121 * pi / 180, 132.5 * pi / 180])
        l6 = DHLink(d=0.080, a=0.0, alpha=0, qlim=[-270 * pi / 180, 270 * pi / 180])

        self.robot = DHRobot([l1, l2, l3, l4, l5, l6], name="Staubli_TX2_60")
        self.standby()
        self.update_base()
        return self.robot

    def standby(self):
        """Return the robot to its default standby joint configuration."""
        self.robot.q = self.q_standby.copy()

    def create_mesh_robot(self, env):
        """Create the cylindrical visual mesh for the robot and add it to the Swift environment."""
        self.update_base()
        self.tx2_60_plot = CylindricalDHRobotPlot(
            self.robot, cylinder_radius=0.05, color="#3478f6"
        ).create_cylinders()
        env.add(self.tx2_60_plot)

    def get_ee_pose(self):
        """Return the current end-effector pose via forward kinematics."""
        return self.robot.fkine(self.robot.q)

    def move_to_q(self, env, q_target, steps=50):
        """Animate a joint-space move, dragging any attached object along with the end-effector."""
        q_traj = jtraj(self.robot.q, q_target, steps).q
        for q in q_traj:
            self.robot.q = q
            if self.attached_object is not None:
                self.attached_object.T = self.get_ee_pose() @ self.grasp_offset
            env.step(0.02)
        return self.robot.q

    def move_to_pose(self, env, target_pose, steps=50):
        """Solve IK for a Cartesian target pose, then animate the move to it."""
        q_sol = self.robot.ikine_LM(target_pose, q0=self.robot.q)
        if not q_sol.success:
            raise RuntimeError(f"IK failed for target pose:\n{target_pose}")
        return self.move_to_q(env, q_sol.q, steps)

    def _downward_pose(self, pose, height=0.0):
        """Make a target pose with the tool's z-axis pointing down, optionally raised vertically."""
        pose = SE3(pose)
        position = pose.t + np.array([0.0, 0.0, height])
        return SE3(position) * SE3.Rx(pi)

    def pick(self, obj, env, approach_height=None, steps=50):
        """Move above the object, descend, grasp it, then retreat to the approach height."""
        obj_pose = SE3(obj.T) if hasattr(obj, "T") else SE3(obj)
        approach_height = self.APPROACH_HEIGHT if approach_height is None else approach_height
        self.move_to_pose(env, self._downward_pose(obj_pose, approach_height), steps)
        self.move_to_pose(env, self._downward_pose(obj_pose), steps)
        self.attached_object = obj
        self.grasp_offset = self.get_ee_pose().inv() * obj_pose
        self.move_to_pose(env, self._downward_pose(obj_pose, approach_height), steps)

    def place(self, bin_pose, env, approach_height=None, steps=50):
        """Move above the target bin, lower the object, release it, then retreat."""
        approach_height = self.APPROACH_HEIGHT if approach_height is None else approach_height
        self.move_to_pose(env, self._downward_pose(bin_pose, approach_height), steps)
        self.move_to_pose(env, self._downward_pose(bin_pose), steps)
        self.attached_object = None
        self.move_to_pose(env, self._downward_pose(bin_pose, approach_height), steps)

    def sort_part(self, env, obj, obj_colour, general_bucket_pose, colour_bucket_poses, steps=50):
        """Pick a part from the known general-bucket pose and place it in its colour bucket."""
        if obj_colour not in self.COLOURS:
            raise ValueError(f"obj_colour must be one of {self.COLOURS}")

        if obj_colour == "red":
            bucket_pose = colour_bucket_poses["red"]
        elif obj_colour == "blue":
            bucket_pose = colour_bucket_poses["blue"]
        else:
            bucket_pose = colour_bucket_poses["green"]

        obj.T = SE3(general_bucket_pose).A
        self.pick(obj, env, steps=steps)
        self.place(bucket_pose, env, steps=steps)
        return obj_colour