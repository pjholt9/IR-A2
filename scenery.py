import swift
from spatialgeometry import Mesh
from spatialmath import SE3
from math import pi
from table import Table
from printer import Printer
from rail import Rail
from linear_platform import Linear_Platform
from EC66 import EC66
import numpy as np

class Scenery:
    def __init__(self):
        self.table = Table(pose=SE3(-0.8, 0.4, 0))
        self.printer1 = Printer(pose=self.table.pose @ SE3(1.575, -0.35, 0.75) @ SE3.Rz(pi))
        self.printer2 = Printer(pose=self.table.pose @ SE3(1.045, -0.35, 0.75) @ SE3.Rz(pi))
        self.printer3 = Printer(pose=self.table.pose @ SE3(0.515, -0.35, 0.75) @ SE3.Rz(pi))
        self.robot_offset = SE3(0.125, 0.1, 0.05)
        self.platform = Linear_Platform(pose= self.table.pose @ SE3(0.06, -0.18, 0.68))
        self.ec66 = EC66(base=self.platform.pose @ self.robot_offset)
        self.robot = None
        # self.rail = Rail(table_pose=self.table.pose)
        
    def move_platform(self, target_pose, env, steps=300):
        trajectory = self.platform.generate_trajectory(target_pose, steps)
        
        for position in trajectory:
            new_pose = SE3(position)
            self.platform.update_pose(new_pose)
            
            self.ec66.base = self.platform.pose @ self.robot_offset
            self.ec66.update_base()
            
            env.step(0.02)
            
    def home(self, env):
        self.move_platform(self.platform.home_pose, env)
        
    def move1(self, env):
        self.move_platform(self.platform.station1, env)
        
    def move2(self, env):
        self.move_platform(self.platform.station2, env)

    def add_to_env(self, env):
        self.table.add_to_env(env)
        self.printer1.add_to_env(env)
        self.printer2.add_to_env(env)
        self.printer3.add_to_env(env)
        self.platform.add_to_env(env)
        self.robot = self.ec66.create_robot()
        self.robot.q = np.deg2rad([0, -45, -90, 0, 0, 0])
        self.ec66.create_mesh_robot(env)
        # self.rail.add_to_env(env)