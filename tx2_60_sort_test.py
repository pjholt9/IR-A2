import numpy as np
import swift
from spatialgeometry import Cuboid, Sphere
from spatialmath import SE3

from bucket_poses import (
    BLACK_BUCKET_POSE,
    BROWN_BUCKET_POSE,
    GENERAL_BUCKET_POSE,
    IMAGE_COLOUR_BUCKET_POSES,
)
from colour_sort import SimulatedCamera
from tx2_60 import TX2_60


BUCKET_WIDTH = 0.24
BUCKET_WALL_THICKNESS = 0.02
BUCKET_WALL_HEIGHT = 0.15
PART_RADIUS = 0.035
START_Q = np.deg2rad([0, -90, 0, 0, 0, 0])


def add_bucket(env, pose, colour):
    """Add a simple open-top bucket at the part/drop target pose."""
    x, y, target_z = pose.t
    floor_top = target_z - PART_RADIUS
    floor_thickness = 0.02
    floor_center_z = floor_top - floor_thickness / 2
    wall_center_z = floor_top + BUCKET_WALL_HEIGHT / 2
    wall_offset = BUCKET_WIDTH / 2 - BUCKET_WALL_THICKNESS / 2

    pieces = [
        Cuboid(
            scale=[BUCKET_WIDTH, BUCKET_WIDTH, floor_thickness],
            pose=SE3(x, y, floor_center_z),
            color=colour,
        ),
        Cuboid(
            scale=[BUCKET_WIDTH, BUCKET_WALL_THICKNESS, BUCKET_WALL_HEIGHT],
            pose=SE3(x, y - wall_offset, wall_center_z),
            color=colour,
        ),
        Cuboid(
            scale=[BUCKET_WIDTH, BUCKET_WALL_THICKNESS, BUCKET_WALL_HEIGHT],
            pose=SE3(x, y + wall_offset, wall_center_z),
            color=colour,
        ),
        Cuboid(
            scale=[BUCKET_WALL_THICKNESS, BUCKET_WIDTH, BUCKET_WALL_HEIGHT],
            pose=SE3(x - wall_offset, y, wall_center_z),
            color=colour,
        ),
        Cuboid(
            scale=[BUCKET_WALL_THICKNESS, BUCKET_WIDTH, BUCKET_WALL_HEIGHT],
            pose=SE3(x + wall_offset, y, wall_center_z),
            color=colour,
        ),
    ]
    for piece in pieces:
        env.add(piece)


def main():
    env = swift.Swift()
    env.launch(realtime=True)
    env.set_camera_pose([1.5, -1.8, 1.5], [0.4, 0.0, 0.2])

    tx2_60 = TX2_60()
    tx2_60.q_standby = START_Q.copy()
    tx2_60.create_robot()
    tx2_60.create_mesh_robot(env)

    add_bucket(env, GENERAL_BUCKET_POSE, "dimgray")
    add_bucket(env, BLACK_BUCKET_POSE, "black")
    add_bucket(env, BROWN_BUCKET_POSE, "saddlebrown")

    camera = SimulatedCamera()

    # Keep the camera window responsive while the arm animates
    swift_step = env.step

    def step_and_pump(*args, **kwargs):
        swift_step(*args, **kwargs)
        camera.pump()

    env.step = step_and_pump

    def spawn_part():
        """Raise the object flag, show/classify a random camera image, and queue a part of that colour."""
        camera.object_present = True
        obj_colour = camera.detect_colour()
        camera.object_present = False
        part = Sphere(
            radius=PART_RADIUS,
            pose=GENERAL_BUCKET_POSE,
            color=camera.last_rgb,
        )
        env.add(part)
        print(f"Detected part colour: {obj_colour}")
        tx2_60.queue_part(part, obj_colour)

    def on_sorted(obj, obj_colour):
        """Close the image once the part is in its bucket, then bring in the next part."""
        camera.close_image()
        spawn_part()

    spawn_part()

    tx2_60.run(
        env,
        GENERAL_BUCKET_POSE,
        IMAGE_COLOUR_BUCKET_POSES,
        on_sorted=on_sorted,
    )


if __name__ == "__main__":
    main()
