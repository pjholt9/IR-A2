"""Hazard demos for the workcell: a moving apple the TX2-60 must avoid, and a person who trips a light curtain.

Both are driven by update(dt), called from the simulation loop on the main thread.
"""

import numpy as np
from ir_support_extra_parts.parts import part_mesh
from spatialgeometry import Cuboid
from spatialmath import SE3

# Shared staging spot keeps inactive hazard meshes out of the workcell view.
PARKED = (100.0, 100.0, -100.0)  # where inactive meshes are hidden
PERSON_FORWARD_YAW = 0.0  # extra yaw if the person mesh's front isn't along its +x axis

# Apple sweeps back and forth across the arm's travel path (table coordinates)
APPLE_SWEEP_X = 0.45
APPLE_Y = 0.2
APPLE_HEIGHT = 0.5 + 0.25  # table top + roughly the height the gripper travels at
APPLE_PERIOD = 8.0  # seconds per full back-and-forth cycle
APPLE_SIZE = 0.12  # collision box the arm avoids (the mesh's real extent isn't used)

# Visitor walks along the e-stop side of the table (table coordinates)
VISITOR_LANE_X = 1.35  # between the table edge (1.05) and the side fence
VISITOR_START_Y = 2.6  # outside the cell, on the open side
VISITOR_END_Y = 0.2  # how far in they walk
CURTAIN_Y = 0.85  # light curtain line: where the side fence ends and the cell opens
VISITOR_SPEED = 1.2  # m/s
VISITOR_PAUSE = 2.0  # seconds standing inside before walking out


# Add/remove a moving apple from the robot's collision model as it crosses the work area.
class Apple:
    """Moving obstacle. While active, the arm waits for it to clear instead of colliding."""

    # The visible mesh and simple collision box move together; only the box is used for safety checks.
    def __init__(self, env, table_pose, tx2_60):
        self.table_pose = table_pose
        self.tx2_60 = tx2_60
        self.mesh = part_mesh("Apple", pose=SE3(*PARKED))
        self.mesh._collision = False
        env.add(self.mesh)
        # Invisible box used only by the arm's collision check
        self.box = Cuboid(scale=[APPLE_SIZE] * 3, pose=SE3(*PARKED))
        self.active = False
        self.t = 0.0

    # Register the obstacle only while the demo is enabled so normal sorting is unaffected.
    def toggle(self):
        """Add the apple to the scene (and the arm's obstacle list), or take it away again."""
        self.active = not self.active
        if self.active:
            self.t = 0.0
            self.tx2_60.add_obstacle(self.box)
            self.update(0.0)
        else:
            self.tx2_60.obstacles.remove(self.box)
            self._set(SE3(*PARKED))
        return self.active

    # Advance the sweep using elapsed time, rather than tying motion speed to frame rate.
    def update(self, dt):
        if not self.active:
            return
        self.t += dt
        # Cosine gives a smooth reversal at each end of the apple's path.
        x = -APPLE_SWEEP_X * np.cos(2 * np.pi * self.t / APPLE_PERIOD)
        self._set(self.table_pose @ SE3(x, APPLE_Y, APPLE_HEIGHT))

    # Keep the rendered apple and the arm's collision shape at exactly the same pose.
    def _set(self, pose):
        self.mesh.T = pose.A
        self.box.T = pose.A


# Simple visitor state machine for demonstrating the open entrance light curtain.
class Visitor:
    """Person who walks in along the side of the table and trips the light curtain (e-stop)."""

    # Start parked, then move the person mesh along a lane expressed in table coordinates.
    def __init__(self, env, table_pose, estop):
        self.table_pose = table_pose
        self.estop = estop
        self.mesh = part_mesh("personMaleCasual", pose=SE3(*PARKED))
        self.mesh._collision = False
        env.add(self.mesh)
        self.state = "idle"  # idle -> in -> wait -> out -> idle
        self.pos = np.array([VISITOR_LANE_X, VISITOR_START_Y])
        self.timer = 0.0
        self.tripped = False

    # Use the curtain boundary to report whether the visitor is actually inside the cell.
    @property
    def inside(self):
        """True while the person is past the light curtain."""
        return self.state != "idle" and self.pos[1] <= CURTAIN_Y

    # Only start a new walk when idle, so repeated triggers cannot reset someone mid-demo.
    def trigger(self):
        if self.state == "idle":
            self.pos = np.array([VISITOR_LANE_X, VISITOR_START_Y])
            self.state = "in"
            self.tripped = False
            self._place(-np.pi / 2)

    # Each state does one phase of the demo: enter/trip, pause, then walk back out.
    def update(self, dt):
        if self.state == "in":
            if self._walk_to(VISITOR_END_Y, dt, -np.pi / 2):
                self.state, self.timer = "wait", VISITOR_PAUSE
            if self.pos[1] <= CURTAIN_Y and not self.tripped:
                self.tripped = True
                print("Light curtain tripped: person entered the cell")
                self.estop.press()
        elif self.state == "wait":
            self.timer -= dt
            if self.timer <= 0:
                self.state = "out"
        elif self.state == "out":
            if self._walk_to(VISITOR_START_Y, dt, np.pi / 2):
                self.state = "idle"
                self.mesh.T = SE3(*PARKED).A
                print("Person left the cell")

    # Clamp each frame's movement so the person stops on the destination instead of overshooting.
    def _walk_to(self, y, dt, yaw):
        """Step along the lane towards y; returns True once there."""
        step = VISITOR_SPEED * dt
        self.pos[1] += np.clip(y - self.pos[1], -step, step)
        self._place(yaw)
        return abs(self.pos[1] - y) < 1e-9

    # Transform the table-local lane position into the world pose for the person model.
    def _place(self, yaw):
        pose = self.table_pose @ SE3(self.pos[0], self.pos[1], 0) @ SE3.Rz(yaw + PERSON_FORWARD_YAW)
        self.mesh.T = pose.A
