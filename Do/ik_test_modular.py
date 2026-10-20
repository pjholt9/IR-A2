"""Full tested pick sequence + geometry-based pre-drop/drop demonstration.

Place next to robot_tasks.py in Do/. Rename the supplied robot_tasks_drop_integrated.py
as robot_tasks.py first. Uses the saved pre_drop_waypoint.json from teach GUI.
"""
from pathlib import Path
import json
import sys
import numpy as np
import swift
from spatialmath import SE3
from scenery import Scenery
from robot_tasks import RobotTasks
from kinematic_analysis import analyse_full_kinematics
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from PH.scenery_PH import SceneryPH, BUCKET_WIDTH, PART_RADIUS


def main():
    env = swift.Swift()
    env.launch(realtime=True)
    ph = SceneryPH()
    ph.add_to_env(env)
    scene = Scenery()
    scene.add_to_env(env)
    env.step(0.1)
    printer = scene.printer1
    part = scene.parts.get(printer)
    if part is None:
        raise RuntimeError('No part selected in printer 1')
    tasks = RobotTasks(scene, env)
    recorder = tasks.open_door(printer)
    tasks.retract_from_door(printer)
    tasks.move_to_clearance(printer)
    tasks.reorient_for_standby(printer)
    tasks.return_to_standby(printer)
    tasks.move_to_camera_pose(printer)
    tasks.camera_capture_dwell(printer, seconds=3.0)
    tasks.move_to_pre_pick(printer, distance=0.10)
    pick_plan = tasks.plan_part_pick(printer, part)
    if not tasks.execute_part_pick(printer, part, pick_plan):
        raise RuntimeError('Pick attachment failed')

    path = Path(__file__).resolve().parent / 'pre_drop_waypoint.json'
    if not path.exists():
        raise FileNotFoundError(f'Teach and save the pre-drop waypoint first: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    required = ('X', 'Y', 'Z', 'Rx', 'Ry', 'Rz')
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f'Missing pre-drop waypoint values: {missing}')
    rx, ry, rz = (np.deg2rad(float(data[k])) for k in ('Rx', 'Ry', 'Rz'))
    T_pre = (SE3(*(float(data[k])/1000 for k in ('X','Y','Z')))
             @ SE3.Rx(rx) @ SE3.Ry(ry) @ SE3.Rz(rz))
    floor_z = float(ph.general_bucket_pose.t[2] - PART_RADIUS)
    clearance_by_part = data.get('ClearanceByPart', {})
    clearance = (float(clearance_by_part[part.kind]) / 1000
                 if isinstance(clearance_by_part, dict) and part.kind in clearance_by_part
                 else None)
    plan = tasks.plan_part_drop(part, ph.general_bucket_pose, floor_z,
                                bucket_width=BUCKET_WIDTH,
                                clearance_m=clearance,
                                pre_drop_pose=T_pre)
    print('PREVIEW COMPLETE. Executing simulated drop (no collision checking).')
    tasks.execute_part_drop(part, plan, recorder=recorder)
    analyse_full_kinematics(scene.robot, recorder)
    print('Holding at pre-drop. Door still open; close/standby/home not implemented.')
    while True:
        env.step(0.05)

if __name__ == '__main__':
    main()
