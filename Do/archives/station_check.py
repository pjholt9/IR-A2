import numpy as np
from scenery import Scenery

scenery = Scenery()

for i, printer in enumerate(scenery.printers, start=1):
    station = scenery.get_printer_station(printer)

    T_robot = station @ scenery.robot_offset
    T_relative = printer.pose.inv() @ T_robot

    print(f"\nPrinter {i}")
    print("Station:", station.t)
    print("Robot base:", T_robot.t)
    print("Calibration error:",
          np.max(np.abs(
              T_relative.A - scenery.printer_to_robot.A
          )))