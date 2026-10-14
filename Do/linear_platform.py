from spatialgeometry import Mesh
from spatialmath import SE3
from math import pi
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"

class Linear_Platform:
    def __init__(self, pose = SE3()):
        self.home_pose = pose
        self.pose = self.home_pose
        
        # self.station1 = self.home_pose @ SE3(0.42, 0, 0)
        # self.station2 = self.home_pose @ SE3(0.80, 0, 0)
        
        self.platform = Mesh(filename=str(MODEL_DIR / 'Platform.stl'), scale=[0.001, 0.001, 0.001])
        self.platform._collision = False
        self.update_pose(self.home_pose)
        
    def update_pose(self, pose):
        self.pose = pose
        self.platform.T = np.asarray(self.pose.A, dtype=float)
        return self.pose

    def get_station_pose(self, distance):
        return self.home_pose @ SE3(distance, 0, 0)
        
    def generate_trajectory(self, target_pose, steps=100):
        start = self.pose.t
        target = target_pose.t
        
        trajectory = np.linspace(start, target, steps)
        
        return trajectory
        

    def add_to_env(self, env):
        env.add(self.platform)