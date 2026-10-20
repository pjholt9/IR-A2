"""Assemble and run the combined PH and Do workcell."""

import sys
from pathlib import Path

import numpy as np
import swift
from spatialgeometry import Mesh

ROOT_DIR = Path(__file__).resolve().parent
DO_DIR = ROOT_DIR / "Do"
sys.path.insert(0, str(DO_DIR))

from Do.scenery import Scenery as SceneryDo

from PH.colour_sort import SimulatedCamera
from PH.manual_control import run_manual_control
from PH.scenery_PH import SceneryPH


MANUAL_CONTROL = False


def _create_do_scenery():
    """Construct Do scenery with scale compatibility for the installed Mesh class."""
    mesh_init = Mesh.__init__

    def mesh_init_with_vector_scale(self, *args, **kwargs):
        scale = kwargs.get("scale")
        if np.isscalar(scale):
            kwargs["scale"] = [scale, scale, scale]
        mesh_init(self, *args, **kwargs)

    Mesh.__init__ = mesh_init_with_vector_scale
    try:
        return SceneryDo()
    finally:
        Mesh.__init__ = mesh_init


def setup_scene():
    """Create the Swift environment, both scenery sections, and camera callbacks."""
    env = swift.Swift()
    env.launch(realtime=True)

    scenery_ph = SceneryPH()
    scenery_ph.add_to_env(env)
    _create_do_scenery().add_to_env(env)

    camera = SimulatedCamera(
        folder=ROOT_DIR / "PH" / "part_images",
        filenames=("BlackPart.jpg", "BrownPart.jpg"),
    )
    swift_step = env.step

    def step_and_pump_camera(*args, **kwargs):
        swift_step(*args, **kwargs)
        camera.pump()

    env.step = step_and_pump_camera
    env.set_camera_pose([3.0, -3.0, 3.0], [-0.7, 0.85, 0.5])

    def on_part_dropped(part, image_path):
        """Classify a dropped part and enqueue it for sorting."""
        camera.object_present = True
        try:
            colour = camera.detect_colour(image_path=image_path)
        finally:
            camera.object_present = False
        if colour is None:
            raise RuntimeError("Camera did not classify the dropped part.")
        print(f"General-bin camera detected: {colour}")
        scenery_ph.tx2_60.queue_part(part, colour)
        return colour

    def on_part_sorted(_part, _colour):
        camera.close_image()

    return env, scenery_ph, on_part_dropped, on_part_sorted


def main():
    """Run automatic sorting or the optional manual-control interface."""
    env, scenery_ph, _on_part_dropped, on_part_sorted = setup_scene()
    tx2_60 = scenery_ph.tx2_60

    if MANUAL_CONTROL:
        scenery_ph.register_obstacles()
        run_manual_control(tx2_60, env)
        return

    tx2_60.run(
        env,
        scenery_ph.general_bucket_pose,
        scenery_ph.colour_bucket_poses,
        on_sorted=on_part_sorted,
    )
