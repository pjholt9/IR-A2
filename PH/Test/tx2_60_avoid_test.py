import sys

import numpy as np
from spatialgeometry import Cuboid
from spatialmath import SE3

if __package__:
    from ..collision import in_collision
    from ..tx2_60 import TX2_60
else:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from collision import in_collision
    from tx2_60 import TX2_60

START_Q = np.deg2rad([0, -90, 0, 0, 0, 0])
TARGET_Q = np.deg2rad([90, -45, 45, 0, 0, 0])


class FakeEnv:
    """Headless stand-in for Swift so the test runs without a browser."""

    def step(self, dt=0.0):
        pass


def make_scenario(env):
    robot = TX2_60()
    robot.q_standby = START_Q.copy()
    robot.create_robot()

    # Cuboid sweeping back and forth across the arm's swing path
    cuboid = Cuboid(scale=[0.15, 0.15, 0.6], pose=SE3(0.45, 0.475, 0.3), color="red")
    state = {"t": 0.0}

    def update(dt):
        state["t"] += dt
        y = 0.475 + 0.225 * np.sin(2 * np.pi * state["t"] / 6.0)
        cuboid.T = SE3(0.45, y, 0.3).A

    robot.add_obstacle(cuboid, update)
    return robot, cuboid


def run(env, robot):
    log = {"collisions": 0, "waited": False}

    def check(dt):
        if in_collision(robot.robot, robot.robot.q, robot.obstacles, margin=0.0):
            log["collisions"] += 1
        log["waited"] |= robot.waiting

    robot.step_hooks.append(check)
    try:
        robot.move_to_q(env, TARGET_Q, steps=100)
        robot.move_to_q(env, START_Q, steps=100)
    finally:
        robot.step_hooks.remove(check)
    return log


def test_avoids_moving_cuboid():
    env = FakeEnv()
    robot, _ = make_scenario(env)
    log = run(env, robot)
    assert log["collisions"] == 0
    assert np.allclose(robot.robot.q, START_Q)


def test_collision_detected():
    robot = TX2_60()
    robot.create_robot()
    box = Cuboid(scale=[0.2, 0.2, 0.2], pose=SE3(0.3, 0, 0.3))
    assert in_collision(robot.robot, np.zeros(6), [box])
    far = Cuboid(scale=[0.2, 0.2, 0.2], pose=SE3(2, 2, 2))
    assert not in_collision(robot.robot, np.zeros(6), [far])


if __name__ == "__main__":
    if "--headless" in sys.argv:
        test_collision_detected()
        test_avoids_moving_cuboid()
        print("ok")
    else:
        import swift

        env = swift.Swift()
        env.launch(realtime=True)
        env.set_camera_pose([1.5, -1.8, 1.5], [0.4, 0.0, 0.2])
        robot, cuboid = make_scenario(env)
        robot.create_mesh_robot(env)
        env.add(cuboid)
        while True:
            print(run(env, robot))
