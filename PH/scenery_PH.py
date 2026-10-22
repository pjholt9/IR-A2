import numpy as np
from ir_support_extra_parts.parts import part_mesh
from spatialgeometry import Cuboid
from spatialmath import SE3

if __package__:
    from .tx2_60 import TX2_60
else:
    from tx2_60 import TX2_60


# Bucket and part dimensions are in metres, matching the robot's table-local layout.
BUCKET_WIDTH = 0.24
BUCKET_WALL_THICKNESS = 0.02
BUCKET_WALL_HEIGHT = 0.15
FLOOR_THICKNESS = 0.02
TABLE_TOP = 0.5
PART_RADIUS = 0.035
START_Q = np.deg2rad([0, -90, 0, 0, 0, 0])
# Only the colours the camera and part models currently support get a bucket
COLOURS = ("black", "brown")
DRAW_COLOURS = {
    "black": "black",
    "brown": "saddlebrown",
}

# These poses describe the workcell relative to the table, then place that table in the world.
ROBOT_LOCAL = SE3(0.0, 0.0, TABLE_TOP)
GENERAL_LOCAL = SE3(0.0, -0.5, TABLE_TOP + FLOOR_THICKNESS + PART_RADIUS)
ARC_RADIUS = 0.5
ARC_CENTRE = np.deg2rad(90)
ARC_HALF_SPAN = np.deg2rad(28)
# E-stop in the table corner on the bucket (+y) side, beyond the arm's ~0.95 m reach
ESTOP_LOCAL_XY = (0.9, 0.58)
ESTOP_SIZE = 0.12  # approximate collision box for the e-stop mesh
# Fence coordinates leave the +y entrance open for the person/light-curtain demo.
FENCE_SIDE_X = (-1.25, 1.75)
FENCE_SIDE_Y = (-1.4, 0.1)
FENCE_BACK_X = (-0.49, 1.03)
FENCE_BACK_Y = -2.15
FENCE_OPENING_Y = 0.85
GENERAL_WORLD_XY = (-0.6, 0.85)
PH_WORKCELL_OFFSET = SE3(0.7, 0.0, 0.0)
TABLE_POSE = PH_WORKCELL_OFFSET @ SE3(
    GENERAL_WORLD_XY[0] - GENERAL_LOCAL.t[0],
    GENERAL_WORLD_XY[1] - GENERAL_LOCAL.t[1],
    0,
)


# Build the shared table scene and compute where the robot, general bin and colour bins sit on it.
class SceneryPH:
    """Set up the TX2-60 sorting station on the IR support table model."""

    # Keep scene construction separate from adding objects to Swift so poses can be inspected first.
    def __init__(self, table_pose=TABLE_POSE, tx2_60=None):
        self.table_pose = SE3(table_pose)
        self.table = part_mesh(
            "tableBrown2.1x1.4x0.5m",
            pose=self.table_pose,
        )
        # The side and rear panels cover both table footprints; the +y side stays open for bucket and e-stop access.
        self.fences = [
            part_mesh(
                "barrier1.5x0.2x1m",
                pose=self.table_pose @ SE3(x, y, 0) @ SE3.Rz(np.pi / 2),
            )
            for x in FENCE_SIDE_X
            for y in FENCE_SIDE_Y
        ]
        self.fences.extend(
            part_mesh(
                "barrier1.5x0.2x1m",
                pose=self.table_pose @ SE3(x, FENCE_BACK_Y, 0),
            )
            for x in FENCE_BACK_X
        )
        self.fences.append(
            part_mesh(
                "SafetyGate",
                pose=self.table_pose
                @ SE3(FENCE_SIDE_X[0] - 0.527, FENCE_OPENING_Y, 0),
            )
        )
        # Static visual meshes
        self.table._collision = False

        for fence in self.fences:
            fence._collision = False
        # Reuse a supplied robot when integrating elsewhere; otherwise mount a new one at this table's robot pose.
        self.tx2_60 = tx2_60 if tx2_60 is not None else TX2_60(
            base=self.table_pose @ ROBOT_LOCAL
        )
        self.robot = None
        self.buckets = {}
        self.estop = None

        self.general_bucket_pose = self.table_pose @ GENERAL_LOCAL
        # Spread the colour bins over a small arc in front of the robot for reachable drop targets.
        angles = np.linspace(
            ARC_CENTRE - ARC_HALF_SPAN,
            ARC_CENTRE + ARC_HALF_SPAN,
            len(COLOURS),
        )
        self.colour_bucket_poses = {
            colour: (
                self.table_pose
                @ ROBOT_LOCAL
                @ SE3.Rz(angle)
                @ SE3(ARC_RADIUS, 0, FLOOR_THICKNESS + PART_RADIUS)
                @ SE3.Rz(-angle)
            )
            for colour, angle in zip(COLOURS, angles)
        }

    # Put the visual workcell into Swift and build the matching robot and bucket meshes.
    def add_to_env(self, env):
        """Add the table, fences, TX2-60, and sorting buckets to a Swift environment."""
        env.add(self.table)
        for fence in self.fences:
            env.add(fence)
        self.tx2_60.base = self.table_pose @ ROBOT_LOCAL
        self.tx2_60.q_standby = START_Q.copy()
        self.robot = self.tx2_60.create_robot()
        self.tx2_60.create_mesh_robot(env)

        self.buckets["general"] = self._add_bucket(
            env, self.general_bucket_pose, "dimgray"
        )
        for colour, pose in self.colour_bucket_poses.items():
            self.buckets[colour] = self._add_bucket(
                env, pose, DRAW_COLOURS[colour]
            )
        self.estop = part_mesh(
            "emergencyStopButton",
            pose=self.table_pose @ SE3(*ESTOP_LOCAL_XY, TABLE_TOP),
        )
        self.estop._collision = False
        env.add(self.estop)
        return self.tx2_60

    # Collision-only geometry gives the arm planner simple shapes instead of testing detailed meshes.
    def register_obstacles(self):
        """Register the table top, buckets and e-stop as collision obstacles for the arm."""
        # Invisible slabs/boxes: only used by the ellipsoid collision check, never added to Swift
        # Top sits 3 cm below the surface so the base column's own ellipsoid tip isn't flagged
        table_top = Cuboid(
            scale=[2.1, 1.4, 0.04],
            pose=self.table_pose @ SE3(0, 0, TABLE_TOP - 0.05),
        )
        estop_box = Cuboid(
            scale=[ESTOP_SIZE, ESTOP_SIZE, ESTOP_SIZE],
            pose=self.table_pose @ SE3(*ESTOP_LOCAL_XY, TABLE_TOP + ESTOP_SIZE / 2),
        )
        for obstacle in [table_top, estop_box]:
            self.tx2_60.add_obstacle(obstacle)
        for pieces in self.buckets.values():
            for piece in pieces:
                self.tx2_60.add_obstacle(piece)

    # Build a bucket around a target pose that represents where a resting part's centre should be.
    @staticmethod
    def _add_bucket(env, pose, colour):
        """Add an open-top bucket whose target pose is the resting part centre."""
        wall_offset = BUCKET_WIDTH / 2 - BUCKET_WALL_THICKNESS / 2
        floor_center = SE3(0, 0, -PART_RADIUS - FLOOR_THICKNESS / 2)
        wall_center_z = -PART_RADIUS + BUCKET_WALL_HEIGHT / 2
        # The floor sits below the part centre and the walls line up around the open top.
        pieces = [
            Cuboid(
                scale=[BUCKET_WIDTH, BUCKET_WIDTH, FLOOR_THICKNESS],
                pose=pose @ floor_center,
                color=colour,
            ),
            Cuboid(
                scale=[BUCKET_WIDTH, BUCKET_WALL_THICKNESS, BUCKET_WALL_HEIGHT],
                pose=pose @ SE3(0, -wall_offset, wall_center_z),
                color=colour,
            ),
            Cuboid(
                scale=[BUCKET_WIDTH, BUCKET_WALL_THICKNESS, BUCKET_WALL_HEIGHT],
                pose=pose @ SE3(0, wall_offset, wall_center_z),
                color=colour,
            ),
            Cuboid(
                scale=[BUCKET_WALL_THICKNESS, BUCKET_WIDTH, BUCKET_WALL_HEIGHT],
                pose=pose @ SE3(-wall_offset, 0, wall_center_z),
                color=colour,
            ),
            Cuboid(
                scale=[BUCKET_WALL_THICKNESS, BUCKET_WIDTH, BUCKET_WALL_HEIGHT],
                pose=pose @ SE3(wall_offset, 0, wall_center_z),
                color=colour,
            ),
        ]
        for piece in pieces:
            piece._collision = False
            env.add(piece)
        return pieces
