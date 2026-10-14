import swift
import numpy as np
import matplotlib.pyplot as plt

from math import pi
from spatialmath import SE3
from spatialgeometry import Sphere
from roboticstoolbox import jtraj
from matplotlib.widgets import Slider, Button
from scenery import Scenery


# -------------------------------------------------- #
# Initialize Swift
# -------------------------------------------------- #

env = swift.Swift()
env.launch(realtime=True)

scenery = Scenery()
scenery.add_to_env(env)
print("Platform home:", scenery.platform.home_pose.t)
print("Robot offset:", scenery.robot_offset.t)
print("Robot base at home:",
      (scenery.platform.home_pose @ scenery.robot_offset).t)

print("Desired robot base at Printer 1:",
      (scenery.printer1.pose @ scenery.printer_to_robot).t)

print("Scenery loaded")

print(
    "Door marker collision:",
    scenery.printer1.door.handle_marker._collision
)

env.step(1)

# Move linear platform to Printer 1 station
scenery.move_to_printer(scenery.printer1, env)

robot = scenery.robot
door = scenery.printer1.door

robot.q = scenery.ec66.q_standby.copy()
door.update_angle(0)

env.step(0.5)


# -------------------------------------------------- #
# Define existing contact and pre-grasp poses
# -------------------------------------------------- #

T_contact = door.get_handle_pose() @ SE3.Rx(-pi/2)
T_pregrasp = T_contact @ SE3(0, 0, -0.10)

# World-frame position of pre-grasp
pregrasp_pos = T_pregrasp.t.copy()

# Candidate offset relative to pre-grasp, in WORLD XYZ
candidate_offset = np.array([
    -0.15,
     0.00,
     0.05
])

candidate_pos = pregrasp_pos + candidate_offset


# -------------------------------------------------- #
# Create waypoint markers
# -------------------------------------------------- #

candidate_marker = Sphere(
    radius=0.015,
    pose=SE3(*candidate_pos),
    color=[1, 0, 0, 1]
)

pregrasp_marker = Sphere(
    radius=0.012,
    pose=SE3(*pregrasp_pos),
    color=[0, 1, 0, 1]
)

candidate_marker._collision = False
pregrasp_marker._collision = False

env.add(candidate_marker)
env.add(pregrasp_marker)

env.step(0.1)


# -------------------------------------------------- #
# IK helpers
# -------------------------------------------------- #

def candidate_pose(position):
    """
    Keep the same EE orientation as the pre-grasp pose.
    Only change world XYZ position.
    """
    return SE3.Rt(
        T_pregrasp.R,
        position
    )


def solve_candidate(position, q0=None):

    if q0 is None:
        q0 = scenery.ec66.q_standby.copy()

    T_target = candidate_pose(position)

    solution = robot.ikine_LM(
        T_target,
        q0=q0,
        joint_limits=True
    )

    return solution


# -------------------------------------------------- #
# Interactive preview
# -------------------------------------------------- #

last_valid_q = robot.q.copy()


def update_preview(position):

    global last_valid_q

    # Always move marker
    candidate_marker.T = SE3(*position)

    solution = solve_candidate(
        position,
        q0=last_valid_q
    )

    if solution.success:

        # Preview target configuration only.
        # This does NOT validate the path.
        robot.q = solution.q.copy()
        last_valid_q = solution.q.copy()

        status_text.set_text(
            "IK: SUCCESS (collision not checked)"
        )
        status_text.set_color("green")

    else:

        status_text.set_text(
            "IK: FAILED"
        )
        status_text.set_color("red")

    env.step(0.01)
    fig.canvas.draw_idle()


# -------------------------------------------------- #
# Test complete joint-space path
# -------------------------------------------------- #

def test_path(event):

    global last_valid_q

    candidate_pos = np.array([
        pregrasp_pos[0] + slider_x.val,
        pregrasp_pos[1] + slider_y.val,
        pregrasp_pos[2] + slider_z.val
    ])

    q_standby = scenery.ec66.q_standby.copy()

    # First waypoint: pre-pre-grasp
    solution1 = solve_candidate(
        candidate_pos,
        q0=q_standby
    )

    if not solution1.success:
        print("Pre-pre-grasp IK failed from standby seed")
        return

    # Second waypoint: pre-grasp
    solution2 = robot.ikine_LM(
        T_pregrasp,
        q0=solution1.q,
        joint_limits=True
    )

    if not solution2.success:
        print("Pre-grasp IK failed from candidate seed")
        return

    # Reset robot to standby before testing
    robot.q = q_standby.copy()
    env.step(0.2)

    print("\n========== TESTING PATH ==========")

    # Standby -> Pre-pre-grasp
    traj1 = jtraj(
        q_standby,
        solution1.q,
        150
    )

    for q in traj1.q:
        robot.q = q
        env.step(0.02)

    print("Reached Pre-pre-grasp")

    env.step(0.3)

    # Pre-pre-grasp -> Pre-grasp
    traj2 = jtraj(
        solution1.q,
        solution2.q,
        150
    )

    for q in traj2.q:
        robot.q = q
        env.step(0.02)

    print("Reached Pre-grasp")

    print(
        "Pre-pre-grasp world XYZ:",
        np.round(candidate_pos, 4)
    )

    print(
        "Offset from pre-grasp:",
        np.round(candidate_pos - pregrasp_pos, 4)
    )

    print(
        "Joint angles at pre-pre-grasp (deg):",
        np.round(np.rad2deg(solution1.q), 2)
    )

    print("==================================\n")

    last_valid_q = robot.q.copy()


# -------------------------------------------------- #
# Matplotlib sliders
# -------------------------------------------------- #

fig, ax = plt.subplots(figsize=(9, 5))

fig.subplots_adjust(
    left=0.12,
    bottom=0.35
)

ax.axis("off")

ax.set_title(
    "EC66 - Interactive Pre-pre-grasp Explorer",
    fontsize=14
)

ax.text(
    0.5, 0.75,
    "Move X/Y/Z sliders to preview the robot configuration.\n"
    "Press TEST PATH to animate Standby -> Candidate -> Pre-grasp.\n"
    "Observe upper arm / forearm clearance in Swift.",
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=11
)

status_text = ax.text(
    0.5, 0.38,
    "IK: Not tested",
    transform=ax.transAxes,
    ha="center",
    fontsize=12
)

ax_x = fig.add_axes((0.20, 0.24, 0.65, 0.035))
ax_y = fig.add_axes((0.20, 0.18, 0.65, 0.035))
ax_z = fig.add_axes((0.20, 0.12, 0.65, 0.035))

slider_x = Slider(
    ax_x,
    "X (m)",
    -0.30,
    0.30,
    valinit=candidate_offset[0],
    valstep=0.01
)

slider_y = Slider(
    ax_y,
    "Y (m)",
    -0.30,
    0.30,
    valinit=candidate_offset[1],
    valstep=0.01
)

slider_z = Slider(
    ax_z,
    "Z (m)",
    -0.30,
    0.30,
    valinit=candidate_offset[2],
    valstep=0.01
)

ax_button = fig.add_axes((0.35, 0.035, 0.30, 0.055))

button = Button(
    ax_button,
    "TEST PATH"
)


# -------------------------------------------------- #
# Slider callback
# -------------------------------------------------- #

def on_slider_change(value):

    position = pregrasp_pos + np.array([
        slider_x.val,
        slider_y.val,
        slider_z.val
    ])

    update_preview(position)


slider_x.on_changed(on_slider_change)
slider_y.on_changed(on_slider_change)
slider_z.on_changed(on_slider_change)

button.on_clicked(test_path)


# Initial preview
update_preview(candidate_pos)

print("Interactive waypoint explorer ready.")
print("Pre-grasp world XYZ:", np.round(pregrasp_pos, 4))

plt.show()

# Keep Swift running after closing Matplotlib
while True:
    env.step(0.05)