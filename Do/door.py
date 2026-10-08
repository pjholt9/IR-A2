
from spatialgeometry import Mesh
from spatialmath import SE3
from pathlib import Path
from math import pi
import numpy as np
from spatialgeometry import Mesh, Sphere

MODEL_DIR = Path(__file__).resolve().parent / "models"


class Door:
    def __init__(self, hinge_pose=SE3()):
        self.hinge_pose = hinge_pose
        self.angle = 0
        
        self.handle_offset = SE3(0.32, 0.0, 0.193)

        self.door = Mesh(
            filename=str(MODEL_DIR / "Door.stl"),
            scale=0.001,
            color=[0.15, 0.15, 0.15, 0.35]
        )
        
        self.handle_marker = Sphere(radius = 0.01, color = 'red')
        self.handle_marker._collision = False

        self.door._collision = False
        self.update_angle(0)

    def update_angle(self, angle):
        self.angle = angle
        self.door.T = self.hinge_pose @ SE3.Rz(self.angle)
        self.handle_marker.T = self.get_handle_pose()
        return self.door.T

    def open_door(self, env, angle=-pi/2, steps=200):
        trajectory = np.linspace(self.angle, angle, steps)

        for q in trajectory:
            self.update_angle(q)
            env.step(0.02)

    def close_door(self, env, steps=200):
        self.open_door(env, angle=0, steps=steps)
        
    def get_handle_pose(self):
        return self.hinge_pose @ SE3.Rz(self.angle) @ self.handle_offset

    def add_to_env(self, env):
        env.add(self.door)
        env.add(self.handle_marker)
