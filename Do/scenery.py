import swift
from spatialgeometry import Mesh
from spatialmath import SE3
from math import pi
from .table import Table
from .printer import Printer
from .rail import Rail
from .linear_platform import Linear_Platform
from .EC66 import EC66
from .build_plate import BuildPlate
from .door import Door
import numpy as np
from spatialgeometry import Sphere

class Scenery:
    def __init__(self):
        self.table = Table(pose=SE3(-0.8, 0.4, 0))
        self.printer1 = Printer(pose=self.table.pose @ SE3(1.575, -0.45, 0.75) @ SE3.Rz(pi))
        self.printer2 = Printer(pose=self.table.pose @ SE3(1.045, -0.45, 0.75) @ SE3.Rz(pi))
        # self.printer3 = Printer(pose=self.table.pose @ SE3(0.515, -0.35, 0.75) @ SE3.Rz(pi))
        self.robot_offset = SE3(0.125, 0.10, 0.05) @ SE3.Rz(pi/2)
        self.platform = Linear_Platform(pose= self.table.pose @ SE3(0.06, -0.18, 0.68))
        self.ec66 = EC66(base=self.platform.pose @ self.robot_offset)
        # self.reference_printer = self.printer1
        # self.reference_x = 0.80
        # T_robot_ref = self.platform.get_station_pose(self.reference_x) @ self.robot_offset
        # self.printer_to_robot = self.reference_printer.pose.inv() @ T_robot_ref
        self.printers = [self.printer1, self.printer2]
        # Build plate origin markers
        self.plate_markers = []

        for printer in self.printers:
            marker = Sphere(
                radius=0.012,
                pose=printer.plate_reference,
                color=[0, 1, 0, 1]
            )
            marker._collision = False

            self.plate_markers.append(marker)
        self.robot = None
        
        self.printer_to_robot = (SE3(0.590, -0.370, -0.020) @ SE3.Rz(-pi/2))
        
        self.waypoint_offset = np.array([
            -0.15,   # X relative to pre-grasp (m)
            0.00,   # Y relative to pre-grasp (m)
            0.05    # Z relative to pre-grasp (m)
        ])

        # self.rail = Rail(table_pose=self.table.pose)
    
    def update_waypoint(self, position):
        """
        Update the waypoint marker using world XYZ coordinates.
        """
        self.waypoint_marker.T = SE3(
            float(position[0]),
            float(position[1]),
            float(position[2])
        )    
    
    def move_platform(self, target_pose, env, steps=300):
        trajectory = self.platform.generate_trajectory(target_pose, steps)

        for position in trajectory:
            new_pose = SE3.Rt(self.platform.home_pose.R, np.asarray(position, dtype=float))

            self.platform.update_pose(new_pose)

            self.ec66.base = self.platform.pose @ self.robot_offset
            self.ec66.update_base()

            env.step(0.02)
            
    def home(self, env):
        self.move_platform(self.platform.home_pose, env)
        
    def move1(self, env):
        self.move_to_printer(self.printer1, env)

    def move2(self, env):
        self.move_to_printer(self.printer2, env)

    def add_to_env(self, env):
        self.table.add_to_env(env)
        self.printer1.add_to_env(env)
        self.printer2.add_to_env(env)
        # self.printer3.add_to_env(env)
        self.platform.add_to_env(env)
        self.robot = self.ec66.create_robot()
        self.ec66.create_mesh_robot(env)
        # self.rail.add_to_env(env)
        for marker in self.plate_markers:
            env.add(marker)
                
    def get_printer_station(self, printer):
        T_robot_target = printer.pose @ self.printer_to_robot
        T_platform_target = T_robot_target @ self.robot_offset.inv()
        T_relative = self.platform.home_pose.inv() @ T_platform_target
        distance = float(T_relative.t[0])

        # Reject printer placements that cannot be served
        # by a 1-DOF rail at this mounting configuration.
        if not np.allclose(T_relative.t[1:], 0.0, atol=1e-4):
            raise ValueError(
                "Printer station is not aligned with the rail."
            )

        if not np.allclose(T_relative.R, np.eye(3), atol=1e-4):
            raise ValueError(
                "Printer orientation is incompatible with "
                "the calibrated rail station."
            )

        return self.platform.get_station_pose(distance)
    
    def move_to_printer(self, printer, env):
        target_pose = self.get_printer_station(printer)
        self.move_platform(target_pose, env)