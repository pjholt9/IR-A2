from spatialgeometry import Mesh
from spatialmath import SE3
from math import pi
from pathlib import Path
from door import Door
from build_plate import BuildPlate


MODEL_DIR = Path(__file__).resolve().parent / "models"

class Printer:
    def __init__(self, pose = SE3()):
        self.pose = pose
        self.body = Mesh(filename=str(MODEL_DIR / 'P1S_Correct.stl'), scale=0.001)
        self.door = Door(hinge_pose=self.pose @ SE3(0.035, 0.004, 0.03))
        self.build_plate = BuildPlate(pose=self.pose @ SE3(0.065, 0.03, 0.25))
        self.body._collision = False
        self.update_pose()
    
    def update_pose(self):
        self.body.T = self.pose
        
    def add_to_env(self, env):
        env.add(self.body)
        self.door.add_to_env(env)
        self.build_plate.add_to_env(env)