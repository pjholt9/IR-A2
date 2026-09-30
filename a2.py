import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import swift
from ir_support_extra_parts.parts import part_names, part_mesh
from ir_support_extra_robots.robots import Turtlebot3Waffle
import roboticstoolbox as rtb
from spatialmath.base import *
from spatialmath import SE3
from spatialgeometry import Sphere, Arrow, Mesh
from roboticstoolbox import DHLink, DHRobot, models
from ir_support import CylindricalDHRobotPlot 
from ir_support.robots import UR3e
from scenery import Scenery
from tx2_60 import TX2_60
import os
from math import pi

# World-frame part and drop poses in metres; adjust to match the scene.
GENERAL_BUCKET_POSE = SE3(0.55, 0.0, 0.10)
GREEN_BUCKET_POSE = SE3(0.25, 0.40, 0.10)
RED_BUCKET_POSE = SE3(0.25, 0.0, 0.10)
BLUE_BUCKET_POSE = SE3(0.25, -0.40, 0.10)
OBJ_COLOUR = "red"
COLOUR_BUCKET_POSES = {
    "green": GREEN_BUCKET_POSE,
    "red": RED_BUCKET_POSE,
    "blue": BLUE_BUCKET_POSE,
}

#************************SCENE SETUP******************************
def setup_scene():
    env = swift.Swift()
    env.launch(realtime=True)

    tx2_60 = TX2_60()
    tx2_60.create_robot()
    tx2_60.create_mesh_robot(env)

    env.set_camera_pose([3.0, -2.0, 2.2], [0.4, 0.2, 0.3])
    return env, tx2_60


if __name__ == "__main__":
    env = swift.Swift()
    env.launch(realtime=True)
    
    scenery = Scenery()
    scenery.add_to_env(env)
    
    env.step(1)
    
    scenery.move1(env)
    env.step(1)
    
    scenery.move2(env)
    env.step(1)
    
    scenery.home(env)
    
    while True:
        env.step(0.05)