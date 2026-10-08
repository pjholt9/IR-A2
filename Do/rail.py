from spatialgeometry import Mesh
from spatialmath import SE3
from math import pi
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"

class Rail:
    def __init__(self, table_pose = SE3()):
        self.pose = table_pose
        self.rail = Mesh(filename=str(MODEL_DIR / 'Linear Rail.stl'), scale=0.001)
        self.rail._collision = False
        self.rail.T = self.pose @ SE3.Rx(pi/2)  # Rotate the rail mesh to be horizontal
    
    def add_to_env(self, env):
        env.add(self.rail)