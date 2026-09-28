import numpy as np

from math import pi
from spatialmath import SE3
from roboticstoolbox import DHRobot, RevoluteDH, DHLink
from ir_support import CylindricalDHRobotPlot

class EC66:
    def __init__(self, base = SE3()):
        self.robot = None
        self.ec66_plot = None
        
        self.base = base

        self.q_stanby = np.zeros(6)
        
    def update_base(self):
        self.robot.base = self.base
        return self.robot.base
        
    def create_robot(self):
        l1 = DHLink(d = 0.096, a = 0, alpha = -pi/2, qlim = [-2*pi, 2*pi])
        l2 = DHLink(d = 0, a = 0.418, alpha = 0, qlim = [-2*pi, 2*pi])
        l3 = DHLink(d = 0, a = 0.398, alpha = 0, qlim = [-165*pi/180, 165*pi/180])
        l4 = DHLink(d = 0.122, a = 0, alpha = -pi/2, qlim = [-2*pi, 2*pi])
        l5 = DHLink(d = 0.098, a = 0, alpha = -pi/2, qlim = [-2*pi, 2*pi])
        l6 = DHLink(d = 0.089, a = 0, alpha = 0, qlim = [-2*pi, 2*pi])
        
        self.robot = DHRobot([l1, l2, l3, l4, l5, l6], name="PETE")
        self.standby()
        self.update_base()
        return self.robot
    
    def standby(self):
        self.robot.q = self.q_stanby.copy()
        
    def create_mesh_robot(self, env):
        self.update_base()
        self.ec66_plot = CylindricalDHRobotPlot(self.robot, cylinder_radius=0.05, color="#3478f6").create_cylinders()
        env.add(self.ec66_plot)