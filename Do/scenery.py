"""IR-A2 scenery: part geometry/configuration lives in printed_part.py."""
from math import pi
import numpy as np
from spatialmath import SE3
from spatialgeometry import Sphere
from table import Table
from printer import Printer
from linear_platform import Linear_Platform
from EC66 import EC66
from printed_part import PrintedPart, PART_CONFIG, centered_part_offset


class Scenery:
    def __init__(self):
        self.table = Table(pose=SE3(-0.8, 0.4, 0))
        self.printer1 = Printer(pose=self.table.pose @ SE3(1.575, -0.45, 0.75) @ SE3.Rz(pi))
        self.printer2 = Printer(pose=self.table.pose @ SE3(1.045, -0.45, 0.75) @ SE3.Rz(pi))
        self.robot_offset = SE3(0.125, 0.10, 0.05) @ SE3.Rz(pi / 2)
        self.platform = Linear_Platform(pose=self.table.pose @ SE3(0.06, -0.18, 0.68))
        self.ec66 = EC66(base=self.platform.pose @ self.robot_offset)
        self.printers = [self.printer1, self.printer2]

        # Future GUI can select the part independently for each printer.
        # None means the printer currently has no printed part in the scene.
        self.printer_part_types = {self.printer1: "body", self.printer2: None}
        self.parts = {}
        self.pick_markers = {}
        for printer, kind in self.printer_part_types.items():
            if kind is None:
                continue
            if kind not in PART_CONFIG:
                raise ValueError(f"Unknown part type: {kind}")
            part = PrintedPart(
                plate_reference=printer.bambu_origin,
                kind=kind,
                offset=centered_part_offset(kind),
            )
            marker = Sphere(
                radius=0.007,
                pose=part.pick_pose,
                color=PART_CONFIG[kind]["marker_color"],
            )
            marker._collision = False
            self.parts[printer] = part
            self.pick_markers[printer] = marker

        # Preserve existing API used by test/GUI scripts.
        self.part1_type = self.printer_part_types[self.printer1]
        self.part1 = self.parts[self.printer1]
        self.part1_pick_marker = self.pick_markers[self.printer1]

        self.plate_markers = []
        for printer in self.printers:
            marker = Sphere(radius=0.012, pose=printer.plate_reference,
                            color=[0, 1, 0, 1])
            marker._collision = False
            self.plate_markers.append(marker)

        self.robot = None
        self.printer_to_robot = SE3(0.590, -0.370, -0.020) @ SE3.Rz(-pi / 2)
        self.waypoint_offset = np.array([-0.15, 0.0, 0.05])

    def update_waypoint(self, position):
        self.waypoint_marker.T = SE3(float(position[0]), float(position[1]),
                                     float(position[2]))

    def move_platform(self, target_pose, env, steps=300):
        trajectory = self.platform.generate_trajectory(target_pose, steps)
        for position in trajectory:
            new_pose = SE3.Rt(self.platform.home_pose.R,
                              np.asarray(position, dtype=float))
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
        self.platform.add_to_env(env)
        self.robot = self.ec66.create_robot()
        self.ec66.create_mesh_robot(env)
        for marker in self.plate_markers:
            env.add(marker)
        for part in self.parts.values():
            part.add_to_env(env)
        for marker in self.pick_markers.values():
            env.add(marker)

    def get_printer_station(self, printer):
        T_robot_target = printer.pose @ self.printer_to_robot
        T_platform_target = T_robot_target @ self.robot_offset.inv()
        T_relative = self.platform.home_pose.inv() @ T_platform_target
        distance = float(T_relative.t[0])
        if not np.allclose(T_relative.t[1:], 0.0, atol=1e-4):
            raise ValueError("Printer station is not aligned with the rail.")
        if not np.allclose(T_relative.R, np.eye(3), atol=1e-4):
            raise ValueError("Printer orientation is incompatible with the calibrated rail station.")
        return self.platform.get_station_pose(distance)

    def move_to_printer(self, printer, env):
        target_pose = self.get_printer_station(printer)
        self.move_platform(target_pose, env)
