from spatialgeometry import Mesh
from spatialmath import SE3
from math import pi
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"

class Table:
    def __init__(self, pose = SE3()):
        self.pose = pose
        self.mesh = Mesh(filename=str(MODEL_DIR / 'Table_Rail.STL'), scale=0.001)
        self.mesh._collision = False
        self.mesh.T = self.pose @ SE3.Rx(pi/2)  # Rotate the table mesh to be horizontal
        
    def add_to_env(self, env):
        env.add(self.mesh)