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
from Do.scenery import Scenery as SceneryDo
from scenery_PH import SceneryPH
import os
from math import pi

#************************SCENE SETUP******************************
def setup_scene():
    env = swift.Swift()
    env.launch(realtime=True)

    scenery_PH = SceneryPH()
    scenery_PH.add_to_env(env)

    scenery_Do = SceneryDo()
    scenery_Do.add_to_env(env)

    env.set_camera_pose([3.0, -2.0, 2.2], [0.4, 0.2, 0.3])
    return env, scenery_PH.tx2_60


if __name__ == "__main__":
    env, _ = setup_scene()

    while True:
        env.step(0.05)