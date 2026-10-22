from pathlib import Path

import numpy as np
import swift
from spatialgeometry import Cuboid
from spatialmath import SE3

if __package__:
    from ..parts import PART_COLOURS, make_part
    from ..bucket_poses import (
        BLACK_BUCKET_POSE,
        BROWN_BUCKET_POSE,
        GENERAL_BUCKET_POSE,
        IMAGE_COLOUR_BUCKET_POSES,
    )
    from ..colour_sort import SimulatedCamera
    from ..tx2_60 import TX2_60
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from parts import PART_COLOURS, make_part
    from bucket_poses import (
        BLACK_BUCKET_POSE,
        BROWN_BUCKET_POSE,
        GENERAL_BUCKET_POSE,
        IMAGE_COLOUR_BUCKET_POSES,
    )
    from colour_sort import SimulatedCamera
    from tx2_60 import TX2_60


# Build the open-top bucket from cuboids so the part can be visibly dropped inside.
BUCKET_WIDTH = 0.24
BUCKET_WALL_THICKNESS = 0.02
BUCKET_WALL_HEIGHT = 0.15
PART_RADIUS = 0.035
START_Q = np.deg2rad([0, -90, 0, 0, 0, 0])


# Add a floor and four walls around the target pose while leaving the top clear.
def add_bucket(env, pose, colour):
    """Add a simple open-top bucket at the part/drop target pose."""
    x, y, target_z = pose.t
    floor_top = target_z - PART_RADIUS  # Leave the floor just below the part's centre at its target pose.
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


# Set up the arm, buckets, and simulated camera, then keep supplying parts to the sorter.
def main():
    env = swift.Swift()
    env.launch(realtime=True)
    env.set_camera_pose([1.5, -1.8, 1.5], [0.4, 0.0, 0.2])

    tx2_60 = TX2_60()
    tx2_60.q_standby = START_Q.copy()
    tx2_60.create_robot()
    tx2_60.create_mesh_robot(env)

    add_bucket(env, GENERAL_BUCKET_POSE, "dimgray")
    add_bucket(env, BLACK_BUCKET_POSE, PART_COLOURS["black"])
    add_bucket(env, BROWN_BUCKET_POSE, "saddlebrown")

    camera = SimulatedCamera(
        folder=Path(__file__).resolve().parents[1] / "part_images",
        filenames=("BlackPart.jpg", "BrownPart.jpg"),
    )

    # Keep the camera window responsive while the arm animates
    swift_step = env.step

    # Pump the camera on every simulation step so its image window stays responsive during motion.
    def step_and_pump(*args, **kwargs):
        swift_step(*args, **kwargs)
        camera.pump()

    env.step = step_and_pump

    # Simulate a detected part and enqueue it using the colour read from a sample image.
    def spawn_part():
        """Raise the object flag, show/classify a random camera image, and queue a part of that colour."""
        camera.object_present = True
        obj_colour = camera.detect_colour()
        camera.object_present = False
        part = make_part(obj_colour, GENERAL_BUCKET_POSE)
        env.add(part)
        print(f"Detected part colour: {obj_colour}")
        tx2_60.queue_part(part, obj_colour)

    # Close the previous image and kick off another sort once the current part is placed.
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
