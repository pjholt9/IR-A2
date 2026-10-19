"""Sort black/brown parts in the same SceneryPH workcell that the final A2 uses.

Run from the repository root with:
    python -m PH.Test.tx2_60_scene_sort_test
Use --headless for a quick non-visual check (sorts --parts parts then exits).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import swift
from ir_support_extra_parts.parts import part_mesh
from spatialmath import SE3

if __package__:
    from ..colour_sort import SimulatedCamera
    from ..manual_control import run_manual_control
    from ..parts import make_part
    from ..safety import EStop
    from ..scenery_PH import TABLE_TOP, SceneryPH
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from colour_sort import SimulatedCamera
    from manual_control import run_manual_control
    from parts import make_part
    from safety import EStop
    from scenery_PH import TABLE_TOP, SceneryPH

# Set True to drive the joints by hand from a slider window instead of auto-sorting
MANUAL_CONTROL = False

# Set True to open a separate E-STOP window that halts the arm during automatic running
ESTOP_GUI = True

# Set True to have an apple fly across the workcell, so the arm has to wait for it to clear
APPLE_OBSTACLE = True
APPLE_SPAWN_TIME = 10.0  # simulation seconds before the apple appears
APPLE_SWEEP_X = 0.45  # half-width of the apple's back-and-forth path, in table coordinates
APPLE_Y = 0.2  # between the arm and the colour buckets, across the arm's travel path
APPLE_HEIGHT = TABLE_TOP + 0.25  # roughly the height the gripper travels at
APPLE_PERIOD = 8.0  # seconds per full back-and-forth cycle




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--parts", type=int, default=2, help="parts to sort in --headless mode")
    args = parser.parse_args()

    env = swift.Swift()
    env.launch(realtime=not args.headless, headless=args.headless)

    # Same table, robot and buckets as A2
    scenery = SceneryPH()
    tx2_60 = scenery.add_to_env(env)
    env.set_camera_pose([1.5, -0.2, 1.4], [-0.6, 0.85, 0.6])

    camera = SimulatedCamera(
        folder=Path(__file__).resolve().parents[1] / "part_images",
        filenames=("BlackPart.jpg", "BrownPart.jpg"),
    )

    # Keep the camera window responsive while the arm animates; also drives the apple's clock
    estop = None
    if ESTOP_GUI and not args.headless and not MANUAL_CONTROL:
        estop = EStop()
        tx2_60.safety = estop
    swift_step = env.step
    apple = {"t": 0.0, "mesh": None}

    def apple_pose():
        """Apple position: sweeps back and forth along table-x, starting at one end."""
        phase = 2 * np.pi * (apple["t"] - APPLE_SPAWN_TIME) / APPLE_PERIOD
        x = -APPLE_SWEEP_X * np.cos(phase)
        return scenery.table_pose @ SE3(x, APPLE_Y, APPLE_HEIGHT)

    def update_apple(dt):
        apple["t"] += dt
        if apple["t"] < APPLE_SPAWN_TIME:
            return
        if apple["mesh"] is None:
            # Added to Swift only: the arm is NOT told about it and must sense it itself
            apple["mesh"] = part_mesh("Apple", pose=apple_pose())
            env.add(apple["mesh"])
            print(f"Apple entered the scene at t={apple['t']:.1f} s")
        apple["mesh"].T = apple_pose().A

    def step_and_pump(*args_, **kwargs):
        swift_step(*args_, **kwargs)
        camera.pump()
        if estop is not None:
            estop.pump()
        if APPLE_OBSTACLE:
            # Every env.step (moving, waiting or idle) advances the simulation clock
            dt = args_[0] if args_ else kwargs.get("dt", 0.05)
            update_apple(dt)

    env.step = step_and_pump
    if APPLE_OBSTACLE:
        # Everything in the scene at this point is known; anything added later is an intruder
        tx2_60.enable_sensing(env)

    sorted_parts = []

    def spawn_part():
        """Stand in for the teammate's drop: classify a random image and queue a part of that colour."""
        camera.object_present = True
        colour = camera.detect_colour()
        camera.object_present = False
        part = make_part(colour, scenery.general_bucket_pose)
        env.add(part)
        print(f"Detected part colour: {colour}")
        tx2_60.queue_part(part, colour)

    def on_sorted(part, colour):
        camera.close_image()
        sorted_parts.append((part, colour))
        # Check the part ended inside its own colour bucket
        bucket = scenery.colour_bucket_poses[colour].t
        distance = abs(part.T[:3, 3][:2] - bucket[:2]).max()
        if distance > 0.05:
            raise AssertionError(f"{colour} part is {distance:.3f} m from its bucket")
        print(f"Sorted {colour} part into its bucket")
        if args.headless and len(sorted_parts) >= args.parts:
            raise SystemExit(0)
        spawn_part()

    if MANUAL_CONTROL:
        scenery.register_obstacles()
        run_manual_control(tx2_60, env)
        return

    spawn_part()
    tx2_60.run(
        env,
        scenery.general_bucket_pose,
        scenery.colour_bucket_poses,
        on_sorted=on_sorted,
    )


if __name__ == "__main__":
    main()
