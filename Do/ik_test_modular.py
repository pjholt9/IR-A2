"""EC66 door-opening demonstration using reusable task and analysis modules."""
import swift
from scenery import Scenery
from robot_tasks import RobotTasks
from kinematic_analysis import analyse_full_kinematics


def main():
    env = swift.Swift()
    env.launch(realtime=True)
    scenery = Scenery()
    scenery.add_to_env(env)
    env.step(0.1)

    # Switch between scenery.printer1 and scenery.printer2.
    target_printer = scenery.printer1
    tasks = RobotTasks(scenery, env)
    recorder = tasks.open_door(target_printer)
    
    tasks.retract_from_door(target_printer)
    tasks.move_to_clearance(target_printer)
    tasks.reorient_for_standby(target_printer)
    tasks.return_to_standby(target_printer)
    tasks.move_to_camera_pose(target_printer)
    tasks.camera_capture_dwell(target_printer, seconds=3.0)
    T_pre_pick = tasks.move_to_pre_pick(target_printer, distance=0.10)
    plan = tasks.plan_part_pick(scenery.printer1, scenery.part1)
    success = tasks.execute_part_pick(scenery.printer1, scenery.part1, plan)
    
    if not success:
        raise RuntimeError("Pick aborted: attachment failed")

    print("Robot holding extracted part. Ready for pre-drop teaching.")
    
    animation = analyse_full_kinematics(scenery.robot, recorder)

    # Standby return requires a validated retract path to avoid door collision.
    # Keep the door open for the subsequent camera and print-retrieval task.
    while True:
        env.step(0.05)


if __name__ == "__main__":
    main()
