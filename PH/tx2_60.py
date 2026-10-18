from math import pi
from pathlib import Path

import numpy as np
from spatialmath import SE3
from roboticstoolbox import DHRobot, DHLink, jtraj
from ir_support import CylindricalDHRobotPlot
from spatialgeometry import Mesh

if __package__:
    from .collision import in_collision
else:
    from collision import in_collision


class TX2_60:
    COLOURS = ("red", "green", "blue", "black", "brown")
    MESH_COLOR = "#3478f6"
    MESH_FILES = (
        "base_link.STL",
        "one_Link.STL",
        "two_Link.STL",
        "three_Link.STL",
        "four_Link.STL",
        "five_Link.STL",
        "six_Link.STL",
    )
    MESH_OFFSETS = (
        SE3.RPY([-pi / 2, 0, 0], order="zyx"),
        SE3(-0.05, 0, 0),
        SE3(-0.665, -0.05, 0) * SE3.RPY([0, 0, pi / 2], order="zyx"),
        SE3(0.03, -0.02, 0) * SE3.RPY([-pi / 2, 0, pi], order="zyx"),
        SE3(0.03, -0.31, 0.02) * SE3.RPY([pi / 2, 0, pi], order="zyx"),
        SE3(0.03, -0.02, 0) * SE3.RPY([-pi / 2, 0, pi], order="zyx"),
        SE3(0.03, -0.02, -0.01) * SE3.RPY([0, 0, pi], order="zyx"),
    )
    APPROACH_HEIGHT = 0.15
    WAIT_STEP = 0.05
    MAX_WAIT = 60.0

    def __init__(self, base=SE3(), use_meshes=True):
        self.robot = None
        self.tx2_60_plot = None
        self.base = base
        self.use_meshes = use_meshes
        self.using_meshes = False
        self.base_mesh = None
        self.link_meshes = []
        self.q_standby = np.zeros(6)
        # Object currently held by the gripper, and the grasp offset relative to the end-effector
        self.attached_object = None
        self.grasp_offset = SE3()
        # Queue of (obj, obj_colour) tuples awaiting sorting, and a busy flag for run()
        self.pending_parts = []
        self.busy = False
        # Cuboid obstacles checked against link ellipsoids, and callbacks (dt) that animate them
        self.obstacles = []
        self.step_hooks = []
        self.waiting = False

    def add_obstacle(self, cuboid, update=None):
        """Register a cuboid to avoid; update(dt) is an optional callback that moves it each step."""
        self.obstacles.append(cuboid)
        if update is not None:
            self.step_hooks.append(update)

    def update_base(self):
        """Sync the robot's base transform with self.base."""
        self.robot.base = self.base
        if self.base_mesh is not None:
            self.base_mesh.T = self.base * self.MESH_OFFSETS[0]
            self.base_mesh._changed = True
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
        """Add STL link visuals, or the cylinder fallback, to the Swift environment."""
        self.update_base()
        mesh_dir = Path(__file__).resolve().parent / "TX2_60meshes"
        missing = [
            mesh_dir / filename
            for filename in self.MESH_FILES
            if not (mesh_dir / filename).is_file()
        ]

        if self.use_meshes and not missing:
            self.link_meshes = []
            for link_index, filename in enumerate(self.MESH_FILES[1:]):
                mesh = Mesh(
                    str(mesh_dir / filename),
                    pose=self.MESH_OFFSETS[link_index + 1],
                    color=self.MESH_COLOR,
                )
                link = self.robot.links[link_index]
                link.geometry = [*link.geometry, mesh]
                self.link_meshes.append(mesh)

            self.base_mesh = Mesh(
                str(mesh_dir / self.MESH_FILES[0]),
                pose=self.base * self.MESH_OFFSETS[0],
                color=self.MESH_COLOR,
            )
            env.add(self.robot)
            env.add(self.base_mesh)
            self.using_meshes = True
            return

        if self.use_meshes and missing:
            missing_names = ", ".join(path.name for path in missing)
            print(
                "Warning: TX2-60 mesh file(s) missing "
                f"({missing_names}); falling back to cylindrical visuals."
            )

        self.base_mesh = None
        self.link_meshes = []
        self.using_meshes = False
        self.tx2_60_plot = CylindricalDHRobotPlot(
            self.robot, cylinder_radius=0.05, color=self.MESH_COLOR
        ).create_cylinders()
        env.add(self.tx2_60_plot)

    def get_ee_pose(self):
        """Return the current end-effector pose via forward kinematics."""
        return self.robot.fkine(self.robot.q)

    def move_to_q(self, env, q_target, steps=50):
        """Animate a joint-space move, dragging any attached object along with the end-effector."""
        q_traj = jtraj(self.robot.q, q_target, steps).q
        for q in q_traj:
            self._wait_until_clear(env, q)
            self.robot.q = q
            if self.attached_object is not None:
                self.attached_object.T = self.get_ee_pose() @ self.grasp_offset
            env.step(0.02)
            for hook in self.step_hooks:
                hook(0.02)
        return self.robot.q

    def _wait_until_clear(self, env, q):
        """Hold the arm still while moving to q would put a link ellipsoid inside an obstacle."""
        waited = 0.0
        while self.obstacles and in_collision(self.robot, q, self.obstacles):
            if waited >= self.MAX_WAIT:
                raise RuntimeError("Path blocked by obstacle for too long")
            self.waiting = True
            env.step(self.WAIT_STEP)
            for hook in self.step_hooks:
                hook(self.WAIT_STEP)
            waited += self.WAIT_STEP
        self.waiting = False

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
        if obj_colour not in self.COLOURS or obj_colour not in colour_bucket_poses:
            raise ValueError(f"obj_colour must be one of {self.COLOURS} with a bucket pose")

        bucket_pose = colour_bucket_poses[obj_colour]

        obj.T = SE3(general_bucket_pose).A
        self.pick(obj, env, steps=steps)
        self.place(bucket_pose, env, steps=steps)
        return obj_colour

    def queue_part(self, obj, obj_colour):
        """Add a part to the queue of parts awaiting sorting."""
        self.pending_parts.append((obj, obj_colour))

    def run(self, env, general_bucket_pose, colour_bucket_poses, steps=50, idle_step=0.05, on_sorted=None):
        """Continuously sort queued parts, only checking for new work once idle (not mid pick-and-place)."""
        while True:
            if not self.pending_parts:
                env.step(idle_step)
                continue

            self.busy = True
            obj, obj_colour = self.pending_parts.pop(0)
            self.sort_part(env, obj, obj_colour, general_bucket_pose, colour_bucket_poses, steps=steps)
            self.busy = False

            if on_sorted is not None:
                on_sorted(obj, obj_colour)