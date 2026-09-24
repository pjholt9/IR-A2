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
import os
from math import pi

def create_tx2_60():
    # Define the Staubli TX2-60 robot using DH-parameters - Patrick Holt
    L1 = DHLink(d=0.375, a=0.050, alpha=-pi/2, qlim=[-pi, pi])                        # axis 1
    L2 = DHLink(d=0.0,   a=0.290, alpha=0,     qlim=[-127.5*pi/180, 127.5*pi/180])    # axis 2
    L3 = DHLink(d=0.0,   a=0.020, alpha=-pi/2, qlim=[-142.5*pi/180, 142.5*pi/180])    # axis 3
    L4 = DHLink(d=0.310, a=0.0,   alpha=pi/2,  qlim=[-270*pi/180, 270*pi/180])        # axis 4
    L5 = DHLink(d=0.0,   a=0.0,   alpha=-pi/2, qlim=[-121*pi/180, 132.5*pi/180])      # axis 5
    L6 = DHLink(d=0.080, a=0.0,   alpha=0,     qlim=[-270*pi/180, 270*pi/180])        # axis 6

    tx2_60 = DHRobot([L1, L2, L3, L4, L5, L6], name="Staubli_TX2_60")
    return tx2_60




#************************SCENE SETUP******************************
def setup_scene():
    # Initialize the scene with necessary elements
    tx2_60 = create_tx2_60()
    tx2_60 = CylindricalDHRobotPlot(tx2_60, cylinder_radius=0.05, color="#3478f6").create_cylinders() #initial design of robot - USE blender to design better mesh
    tx2_60.q = np.zeros(6)  # Set initial joint configuration to zero

    ur3e = UR3e()  # Initialize the UR3e robot
    ur3e.q = np.zeros(6)  # Set initial joint configuration to zero

    #launch swift and add objects
    env = swift.Swift()
    env.launch(realtime=True)
    env.set_camera_pose([3.0, -2.0, 2.2], [0.4, 0.2, 0.3])  # (position, look-at)
    

    