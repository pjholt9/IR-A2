import numpy as np

from math import pi
from spatialmath import SE3
from roboticstoolbox import DHRobot, DHLink
from ir_support import CylindricalDHRobotPlot


class TX2_60:
    def __init__(self, base=SE3()):
        self.robot = None
        self.tx2_60_plot = None
        self.base = base
        self.q_standby = np.zeros(6)

    def update_base(self):
        self.robot.base = self.base
        return self.robot.base

    def create_robot(self):
        l1 = DHLink(d=0.375, a=0.050, alpha=-pi / 2, qlim=[-pi, pi])
        l2 = DHLink(d=0.0, a=0.290, alpha=0, qlim=[-127.5 * pi / 180, 127.5 * pi / 180])
        l3 = DHLink(d=0.0, a=0.020, alpha=-pi / 2, qlim=[-142.5 * pi / 180, 142.5 * pi / 180])
        l4 = DHLink(d=0.310, a=0.0, alpha=pi / 2, qlim=[-270 * pi / 180, 270 * pi / 180])
        l5 = DHLink(d=0.0, a=0.0, alpha=-pi / 2, qlim=[-121 * pi / 180, 132.5 * pi / 180])
        l6 = DHLink(d=0.080, a=0.0, alpha=0, qlim=[-270 * pi / 180, 270 * pi / 180])

        self.robot = DHRobot([l1, l2, l3, l4, l5, l6], name="Staubli_TX2_60")
        self.standby()
        self.update_base()
        return self.robot

    def standby(self):
        self.robot.q = self.q_standby.copy()

    def create_mesh_robot(self, env):
        self.update_base()
        self.tx2_60_plot = CylindricalDHRobotPlot(
            self.robot, cylinder_radius=0.05, color="#3478f6"
        ).create_cylinders()
        env.add(self.tx2_60_plot)