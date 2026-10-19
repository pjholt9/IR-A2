import swift
from spatialmath import SE3
from math import pi

from table import Table
from printer import Printer
from linear_platform import Linear_Platform


class Scenery:
    def __init__(self):
        # Table
        self.table = Table(
            pose=SE3(-0.8, 0.4, 0)
        )

        # Two 3D printers
        self.printer1 = Printer(
            pose=self.table.pose @ SE3(1.575, -0.45, 0.75) @ SE3.Rz(pi)
        )

        self.printer2 = Printer(
            pose=self.table.pose @ SE3(1.045, -0.45, 0.75) @ SE3.Rz(pi)
        )

        self.printers = [self.printer1, self.printer2]

        # Linear platform (no robot)
        self.platform = Linear_Platform(
            pose=self.table.pose @ SE3(0.06, -0.18, 0.68)
        )


if __name__ == "__main__":
    env = swift.Swift()
    env.launch(realtime=True)

    scenery = Scenery()

    scenery.table.add_to_env(env)
    scenery.printer1.add_to_env(env)
    scenery.printer2.add_to_env(env)
    scenery.platform.add_to_env(env)

    while True:
        env.step(0.05)