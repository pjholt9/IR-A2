import numpy as np
from ir_support_extra_parts.parts import part_mesh
from spatialgeometry import Cuboid
from spatialmath import SE3

if __package__:
    from .tx2_60 import TX2_60
else:
    from tx2_60 import TX2_60


BUCKET_WIDTH = 0.24
BUCKET_WALL_THICKNESS = 0.02
BUCKET_WALL_HEIGHT = 0.15
FLOOR_THICKNESS = 0.02
TABLE_TOP = 0.5
PART_RADIUS = 0.035
START_Q = np.deg2rad([0, -90, 0, 0, 0, 0])
COLOURS = ("black", "blue", "brown", "green", "red")
DRAW_COLOURS = {
    "black": "black",
    "blue": "blue",
    "brown": "saddlebrown",
    "green": "green",
    "red": "red",
}

ROBOT_LOCAL = SE3(0.0, 0.0, TABLE_TOP)
GENERAL_LOCAL = SE3(0.0, -0.5, TABLE_TOP + FLOOR_THICKNESS + PART_RADIUS)
ARC_RADIUS = 0.5
ARC_CENTRE = np.deg2rad(90)
ARC_HALF_SPAN = np.deg2rad(75)
GENERAL_WORLD_XY = (-0.6, 0.85)
TABLE_POSE = SE3(
    GENERAL_WORLD_XY[0] - GENERAL_LOCAL.t[0],
    GENERAL_WORLD_XY[1] - GENERAL_LOCAL.t[1],
    0,
)


class SceneryPH:
    """Set up the TX2-60 sorting station on the IR support table model."""

    def __init__(self, table_pose=TABLE_POSE, tx2_60=None):
        self.table_pose = SE3(table_pose)
        self.table = part_mesh(
            "tableBrown2.1x1.4x0.5m",
            pose=self.table_pose,
        )
        self.tx2_60 = tx2_60 if tx2_60 is not None else TX2_60(
            base=self.table_pose @ ROBOT_LOCAL
        )
        self.robot = None
        self.buckets = {}

        self.general_bucket_pose = self.table_pose @ GENERAL_LOCAL
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

    def add_to_env(self, env):
        """Add the table, TX2-60, and sorting buckets to a Swift environment."""
        env.add(self.table)
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
        return self.tx2_60

    @staticmethod
    def _add_bucket(env, pose, colour):
        """Add an open-top bucket whose target pose is the resting part centre."""
        wall_offset = BUCKET_WIDTH / 2 - BUCKET_WALL_THICKNESS / 2
        floor_center = SE3(0, 0, -PART_RADIUS - FLOOR_THICKNESS / 2)
        wall_center_z = -PART_RADIUS + BUCKET_WALL_HEIGHT / 2
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
            env.add(piece)
        return pieces
