from math import pi
from pathlib import Path

import numpy as np
from spatialmath import SE3
from roboticstoolbox import DHRobot, DHLink, jtraj
from ir_support import CylindricalDHRobotPlot
from spatialgeometry import Cuboid, Mesh, Shape

if __package__:
    from .collision import in_collision
else:
    from collision import in_collision


# i keep the arm model, sorting sequence, and runtime state together here
class TX2_60:
    """Staubli TX2-60 sorting arm.

    Motion pipeline used by the sorting task (pick -> place):
      1. Waypoints: Cartesian poses (above part, at part, above bin, at bin, retreat).
      2. Inverse kinematics (move_to_pose): ikine_LM turns each waypoint into joint angles.
      3. Trajectory generation (move_to_q): jtraj gives a smooth joint-space path between poses.
      4. Collision avoidance (_wait_until_clear): every step is checked against obstacles
         and the arm pauses until the way is clear.
    RMRC (damped least squares, Cartesian straight lines) is demonstrated separately in
    tx2_60_rmrc_test.py, with singularity analysis in tx2_60_kinematic_analysis.py.
    Visual servoing is not implemented: colour_sort.py only classifies the part colour from
    an image, it doesn't feed image error back into the arm's motion.
    """

        # These shared settings keep the sorter and its visual model consistent.
    COLOURS = ("red", "green", "blue", "black", "brown")
    MESH_COLOR = "#ffff00"
    MESH_FILES = (
        "base_link.STL",
        "one_Link.STL",
        "two_Link.STL",
        "three_Link.STL",
        "four_Link.STL",
        "five_Link.STL",
        "six_Link.STL",
    )
        # Each offset places an STL link model onto the matching DH link frame.
    MESH_OFFSETS = (
        SE3.RPY([-pi / 2, 0, 0], order="zyx"),
        SE3(-0.05, 0, 0),
        SE3(-0.665, -0.05, 0) * SE3.RPY([0, 0, pi / 2], order="zyx"),
        SE3.RPY([-pi / 2, 0, pi], order="zyx"),
        SE3(0, -0.31, 0) * SE3.RPY([pi / 2, 0, pi], order="zyx"),
        SE3.RPY([-pi / 2, 0, pi], order="zyx"),
        SE3.RPY([0, 0, pi], order="zyx"),
    )
    APPROACH_HEIGHT = 0.15
    # Flange height above the part centre when gripping, so the part hangs below the tool
    GRASP_HEIGHT = 0.015
    WAIT_STEP = 0.05
    MAX_WAIT = 60.0

        # Set up the robot, workcell bookkeeping, and optional sensing/safety hooks.
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
        # Runtime sensing: unknown objects that appear in the scene become tracked obstacles
        self.sensing = False
        self.known_objects = set()
        self.sensed = {}
        # Optional EStop (PH/safety.py): while it is latched the arm holds still every step
        self.safety = None
        self.sense_size = 0.12

        # Treat the current scene as known so later arrivals can be detected.
    def enable_sensing(self, env, size=0.12):
        """Treat everything in the scene now as known; later arrivals are detected as obstacles.

        Only the object's pose is sensed, so each intruder is modelled as a cube of side
        `size` (the mesh's real extent isn't available to the collision check).
        """
        self.sensing = True
        self.sense_size = size
        self.known_objects = {id(o) for o in env.swift_objects}

        # Track new shapes as simple obstacle proxies and keep their poses up to date.
    def _sense(self, env):
        """Register newly appeared shapes as obstacles and refresh the poses of tracked ones."""
        if not self.sensing:
            return
        for obj in env.swift_objects:
            if id(obj) in self.known_objects or not isinstance(obj, Shape):
                continue
            box = Cuboid(scale=[self.sense_size] * 3, pose=SE3(obj.T))
            self.sensed[id(obj)] = (obj, box)
            self.known_objects.add(id(obj))
            self.obstacles.append(box)
            print("Obstacle detected in the workcell")
        for obj, box in self.sensed.values():
            box.T = obj.T

        # Register a cuboid and, if needed, its animation callback for wait loops.
    def add_obstacle(self, cuboid, update=None):
        """Register a cuboid to avoid; update(dt) is an optional callback that moves it each step."""
        self.obstacles.append(cuboid)
        if update is not None:
            self.step_hooks.append(update)

        # Keep the DH base and visible base mesh aligned in the world.
    def update_base(self):
        """Sync the robot's base transform with self.base."""
        self.robot.base = self.base
        if self.base_mesh is not None:
            self.base_mesh.T = self.base * self.MESH_OFFSETS[0]
            self.base_mesh._changed = True
        return self.robot.base

        # Build the six-link standard-DH model using the arm dimensions and joint limits.
    def create_robot(self):
        """Build the DH model of the Staubli TX2-60 arm."""
                # These DH parameters encode each joint's link offset, length, twist, and travel.
        l1 = DHLink(d=0.375, a=0.050, alpha=-pi / 2, qlim=[-pi, pi])
        l2 = DHLink(d=0.0, a=0.290, alpha=0, qlim=[-127.5 * pi / 180, 127.5 * pi / 180])
        l3 = DHLink(d=0.020, a=0.0, alpha=-pi / 2, qlim=[-142.5 * pi / 180, 142.5 * pi / 180])
        l4 = DHLink(d=0.310, a=0.0, alpha=pi / 2, qlim=[-270 * pi / 180, 270 * pi / 180])
        l5 = DHLink(d=0.0, a=0.0, alpha=-pi / 2, qlim=[-121 * pi / 180, 132.5 * pi / 180])
        l6 = DHLink(d=0.070, a=0.0, alpha=0, qlim=[-270 * pi / 180, 270 * pi / 180])

        self.robot = DHRobot([l1, l2, l3, l4, l5, l6], name="Staubli_TX2_60")
        self.standby()
        self.update_base()
        return self.robot

        # Return to the saved home joint configuration.
    def standby(self):
        """Return the robot to its default standby joint configuration."""
        self.robot.q = self.q_standby.copy()

        # Use detailed STL visuals when available, otherwise draw the cylinder fallback.
    def create_mesh_robot(self, env):
        """Add STL link visuals, or the cylinder fallback, to the Swift environment."""
        self.update_base()
        mesh_dir = Path(__file__).resolve().parent / "TX2_60meshes"
        missing = [ #checks which mesh files are missing
            mesh_dir / filename
            for filename in self.MESH_FILES
            if not (mesh_dir / filename).is_file()
        ]

                # Attach each mesh at its model offset so the visual links line up with the DH frames.
        if self.use_meshes and not missing:
            self.link_meshes = []
            for link_index, filename in enumerate(self.MESH_FILES[1:]):
                mesh = Mesh(
                    str(mesh_dir / filename),
                    pose=self.MESH_OFFSETS[link_index + 1],
                    color=self.MESH_COLOR,
                )
                mesh._collision = False
                link = self.robot.links[link_index]
                link.geometry = [*link.geometry, mesh]
                self.link_meshes.append(mesh)

            self.base_mesh = Mesh(
                str(mesh_dir / self.MESH_FILES[0]),
                pose=self.base * self.MESH_OFFSETS[0],
                color=self.MESH_COLOR,
            )
            self.base_mesh._collision = False
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

        # Read the tool pose from the current joints using forward kinematics.
    def get_ee_pose(self):
        """Return the current end-effector pose via forward kinematics."""
        return self.robot.fkine(self.robot.q)

        # Interpolate a joint-space move and carry any grasped part along with the tool.
    def move_to_q(self, env, q_target, steps=50):
        """Animate a joint-space move, dragging any attached object along with the end-effector."""
        # Trajectory generation: quintic polynomial per joint from the current q to the target,
        # starting and ending at zero velocity (smooth, no jerk at waypoints)
        q_traj = jtraj(self.robot.q, q_target, steps).q
        for q in q_traj:
            # Collision avoidance: check this step before moving, hold still if blocked
            self._wait_until_clear(env, q)
            self._wait_for_safety(env)
            self.robot.q = q
            # Carried part follows the end-effector via the offset recorded at grasp time
            if self.attached_object is not None:
                self.attached_object.T = self.get_ee_pose() @ self.grasp_offset
            env.step(0.02)
            for hook in self.step_hooks:
                hook(0.02)
        return self.robot.q

        # Keep the simulation responsive while the latched e-stop holds the arm still.
    def _wait_for_safety(self, env):
        """Hold the arm still until the e-stop has been reset and restarted."""
        while self.safety is not None and self.safety.stopped:
            env.step(self.WAIT_STEP)
            for hook in self.step_hooks:
                hook(self.WAIT_STEP)

        # Pause before moving if the proposed joint pose overlaps a sampled obstacle.
    def _wait_until_clear(self, env, q):
        """Hold the arm still while moving to q would put a link ellipsoid inside an obstacle."""
        waited = 0.0
        self._sense(env)
        # Reactive avoidance: the arm doesn't replan around the obstacle, it stops and lets
        # moving obstacles (animated by step_hooks) clear, giving up after MAX_WAIT seconds
        while self.obstacles and in_collision(self.robot, q, self.obstacles):
            if waited >= self.MAX_WAIT:
                raise RuntimeError("Path blocked by obstacle for too long")
            self.waiting = True
            env.step(self.WAIT_STEP)
            # Keep obstacles moving while waiting, otherwise they would never clear
            for hook in self.step_hooks:
                hook(self.WAIT_STEP)
            self._sense(env)
            waited += self.WAIT_STEP
        self.waiting = False

        # Solve the Cartesian target with numerical IK, then animate its joint solution.
    def move_to_pose(self, env, target_pose, steps=50):
        """Solve IK for a Cartesian target pose, then animate the move to it."""
        # Inverse kinematics: Levenberg-Marquardt (damped numerical IK), seeded with the current
        # q so the solution stays near the present arm configuration and avoids joint flips
        q_sol = self.robot.ikine_LM(target_pose, q0=self.robot.q)
        if not q_sol.success:
            raise RuntimeError(f"IK failed for target pose:\n{target_pose}")
        return self.move_to_q(env, self._nearest_q(q_sol.q), steps)

    def move_linear(self, env, target_pose, steps=50):
        """Straight-line Cartesian move using resolved-motion rate control (RMRC)."""
        # Start and goal positions of the end-effector
        p0 = self.get_ee_pose().t
        p1 = SE3(target_pose).t
        # Constant Cartesian velocity for a straight line over steps * 0.02 s
        v = (p1 - p0) / (steps * 0.02)
        for i in range(steps):
            # Where the end-effector should be at this step along the line
            desired = p0 + (p1 - p0) * (i + 1) / steps
            # Feedforward velocity plus position feedback; zeros hold the tool orientation
            task_velocity = np.concatenate([v + 5.0 * (desired - self.get_ee_pose().t), [0, 0, 0]])
            # Jacobian and its smallest singular value (small means close to a singularity)
            J = self.robot.jacob0(self.robot.q)
            sigma_min = np.linalg.svd(J, compute_uv=False)[-1]
            # Damped least squares: more damping the closer we are to a singularity
            damping = 0.01 + 1.2 * max(0, 0.08 - sigma_min)
            qdot = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), task_velocity)
            # Integrate the joint velocity over one 0.02 s step
            q_next = self.robot.q + 0.02 * qdot
            # Hold for obstacles and the e-stop before moving, like move_to_q
            self._wait_until_clear(env, q_next)
            self._wait_for_safety(env)
            self.robot.q = q_next
            # A held part follows the end-effector
            if self.attached_object is not None:
                self.attached_object.T = self.get_ee_pose() @ self.grasp_offset
            env.step(0.02)
            for hook in self.step_hooks:
                hook(0.02)

        # Prefer an equivalent joint angle near the current pose without breaking limits.
    def _nearest_q(self, q_target):
        """Shift each joint by whole turns to the equivalent angle closest to the current one."""
        q_now = self.robot.q
        q_near = q_now + (q_target - q_now + pi) % (2 * pi) - pi
        qlim = np.array([link.qlim for link in self.robot.links])
        # Keep the wrapped angle only where it still respects the joint limits
        ok = (q_near >= qlim[:, 0]) & (q_near <= qlim[:, 1])
        return np.where(ok, q_near, q_target)

        # Aim the tool down at a target and choose yaw from its direction relative to the base.
    def _downward_pose(self, pose, height=0.0):
        """Make a target pose with the tool's z-axis pointing down, optionally raised vertically."""
        pose = SE3(pose)
        position = pose.t + np.array([0.0, 0.0, height])
        # Yaw follows the direction from the base, so joint 6 doesn't have to counter-rotate
        # joint 1 as the arm swings between the buckets
        offset = position - self.base.t
        yaw = np.arctan2(offset[1], offset[0])
        # Rx(pi) flips the tool z-axis to point down, so the gripper approaches from above
        return SE3(position) * SE3.Rz(yaw) * SE3.Rx(pi)

        # Approach from above, grasp, then lift so the part clears the surrounding rim.
    def pick(self, obj, env, approach_height=None, steps=50):
        """Move above the object, descend, grasp it, then retreat to the approach height."""
        obj_pose = SE3(obj.T) if hasattr(obj, "T") else SE3(obj)
        approach_height = self.APPROACH_HEIGHT if approach_height is None else approach_height
        # Multi-pose plan: each leg is IK + a joint trajectory, collision-checked per step.
        # Leg 1: go above the part so the final descent is a clean vertical approach
        self.move_to_pose(env, self._downward_pose(obj_pose, self.GRASP_HEIGHT + approach_height), steps)
        # Leg 2: descend onto the part in a straight line
        self.move_linear(env, self._downward_pose(obj_pose, self.GRASP_HEIGHT), steps)
        # Grasp: store the part's pose relative to the gripper so it rides along from now on
        self.attached_object = obj
        self.grasp_offset = self.get_ee_pose().inv() * obj_pose
        # Leg 3: lift straight back up before travelling, so the part clears the bucket rim
        self.move_linear(env, self._downward_pose(obj_pose, self.GRASP_HEIGHT + approach_height), steps)

        # Lower the carried part into the bin, release it, and retreat above the rim.
    def place(self, bin_pose, env, approach_height=None, steps=50):
        """Move above the target bin, lower the object, release it, then retreat."""
        approach_height = self.APPROACH_HEIGHT if approach_height is None else approach_height
        # Travel leg: from the lifted pick pose, a single joint-space move to above the bin
        self.move_to_pose(env, self._downward_pose(bin_pose, self.GRASP_HEIGHT + approach_height), steps)
        # Lower the part into the bin in a straight line
        self.move_linear(env, self._downward_pose(bin_pose, self.GRASP_HEIGHT), steps)
        # Release: the part stays where it was dropped
        self.attached_object = None
        # Retreat upward so the arm leaves clear of the bin
        self.move_linear(env, self._downward_pose(bin_pose, self.GRASP_HEIGHT + approach_height), steps)

        # Choose the colour destination and run the full pick-and-place sequence.
    def sort_part(self, env, obj, obj_colour, general_bucket_pose, colour_bucket_poses, steps=50):
        """Pick a part from the known general-bucket pose and place it in its colour bucket."""
        if obj_colour not in self.COLOURS or obj_colour not in colour_bucket_poses:
            raise ValueError(f"obj_colour must be one of {self.COLOURS} with a bucket pose")

        # The colour (from the camera classifier) selects the destination bucket pose
        bucket_pose = colour_bucket_poses[obj_colour]
        # Parts that grasp above their resting-centre height (grasp_lift) need the bucket target raised
        bucket_pose = SE3(0, 0, getattr(obj, "grasp_lift", 0.0)) * SE3(bucket_pose)

        # Part is assumed to sit at the general bucket pose unless it reports its own pose (keep_pose)
        if not getattr(obj, "keep_pose", False):
            obj.T = SE3(general_bucket_pose).A
        self.pick(obj, env, steps=steps)
        self.place(bucket_pose, env, steps=steps)
        return obj_colour

        # Add a part to the work queue and mark it as ours, not a sensed intruder.
    def queue_part(self, obj, obj_colour):
        """Add a part to the queue of parts awaiting sorting."""
        self.pending_parts.append((obj, obj_colour))
        # Queued parts are the arm's own work, not intruders
        self.known_objects.add(id(obj))

        # Work through queued parts and return to standby between placements.
    def run(self, env, general_bucket_pose, colour_bucket_poses, steps=50, idle_step=0.05, on_sorted=None):
        """Continuously sort queued parts, only checking for new work once idle (not mid pick-and-place)."""
        while True:
            if not self.pending_parts:
                env.step(idle_step)
                continue

            self.busy = True
            obj, obj_colour = self.pending_parts.pop(0)
            self.sort_part(env, obj, obj_colour, general_bucket_pose, colour_bucket_poses, steps=steps)
            # Leave the bucket area so a person can reach in and take the sorted part
            self.move_to_q(env, self._nearest_q(self.q_standby), steps)
            self.busy = False

            if on_sorted is not None:
                on_sorted(obj, obj_colour)
