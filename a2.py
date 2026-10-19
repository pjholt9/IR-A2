import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import swift
from ir_support_extra_parts.parts import part_names, part_mesh
from ir_support_extra_robots.robots import Turtlebot3Waffle
import roboticstoolbox as rtb
from spatialmath.base import *
from spatialmath import SE3
from spatialgeometry import Sphere, Arrow, Mesh
from roboticstoolbox import DHLink, DHRobot, models
from ir_support import CylindricalDHRobotPlot 
from ir_support.robots import UR3e
from Do.scenery import Scenery as SceneryDo
from PH.colour_sort import SimulatedCamera
from PH.scenery_PH import SceneryPH
import os
from math import pi
from pathlib import Path

# Set True to drive the TX2-60 joints by hand from a slider window instead of auto-sorting
MANUAL_CONTROL = False

#************************SCENE SETUP******************************
def setup_scene():
    env = swift.Swift()
    env.launch(realtime=True)

    scenery_PH = SceneryPH()
    scenery_PH.add_to_env(env)

    scenery_Do = SceneryDo()
    scenery_Do.add_to_env(env)

    camera = SimulatedCamera(
        folder=Path(__file__).resolve().parent / "PH" / "part_images",
        filenames=("BlackPart.jpg", "BrownPart.jpg"),
    )
    swift_step = env.step

    def step_and_pump_camera(*args, **kwargs):
        swift_step(*args, **kwargs)
        camera.pump()

    env.step = step_and_pump_camera

    env.set_camera_pose([3.0, -3.0, 3.0], [-0.7, 0.85, 0.5])

    def on_part_dropped(part, image_path):
        """Classify the captured image for a dropped part and enqueue it for sorting."""
        camera.object_present = True
        try:
            colour = camera.detect_colour(image_path=image_path)
        finally:
            camera.object_present = False
        if colour is None:
            raise RuntimeError("Camera did not classify the dropped part.")
        print(f"General-bin camera detected: {colour}")
        scenery_PH.tx2_60.queue_part(part, colour)
        return colour

    def on_part_sorted(_part, _colour):
        camera.close_image()

    return env, scenery_PH, on_part_dropped, on_part_sorted


if __name__ == "__main__":
    env, scenery_PH, on_part_dropped, on_part_sorted = setup_scene()
    tx2_60 = scenery_PH.tx2_60

    if MANUAL_CONTROL:
        scenery_PH.register_obstacles()
        from PH.manual_control import run_manual_control
        run_manual_control(tx2_60, env)
        raise SystemExit(0)

    # Do handoff: after dropping a part in the general bin, call
    # on_part_dropped(part, image_path) with the part object and its captured image.
    # The camera classifies the image and queues the part for the TX2-60.
    tx2_60.run(
        env,
        scenery_PH.general_bucket_pose,
        scenery_PH.colour_bucket_poses,
        on_sorted=on_part_sorted,
    )