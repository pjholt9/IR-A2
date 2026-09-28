from spatialgeometry import Mesh
from spatialmath import SE3
from math import pi

class Printer:
    def __init__(self, pose = SE3()):
        self.pose = pose
        self.body = Mesh(filename='models/P1S_BambuLab.stl', scale=0.001)
        self.body._collision = False
        self.update_pose()
    
    def update_pose(self):
        self.body.T = self.pose
        
    def add_to_env(self, env):
        env.add(self.body)