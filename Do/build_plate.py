
from spatialgeometry import Mesh
from spatialmath import SE3
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"


class BuildPlate:
    def __init__(self, pose=SE3()):
        self.pose = pose

        self.plate = Mesh(
            filename=str(MODEL_DIR / "Build Plate.STL"), scale=0.001, color="#FFD700")

        self.plate._collision = False
        self.update_pose(self.pose)

    def update_pose(self, pose):
        self.pose = pose
        self.plate.T = self.pose
        return self.pose

    def add_to_env(self, env):
        env.add(self.plate)
