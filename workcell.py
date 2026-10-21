"""Assemble and run the combined PH and Do workcell."""

import json
import queue
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import swift
from ir_support_extra_parts.parts import part_mesh
from spatialgeometry import Mesh
from spatialmath import SE3

ROOT_DIR = Path(__file__).resolve().parent
DO_DIR = ROOT_DIR / "Do"
sys.path.insert(0, str(DO_DIR))

from Do.scenery import Scenery as SceneryDo
from printed_part import PART_CONFIG, PrintedPart, centered_part_offset
from robot_tasks import RobotTasks

from PH.colour_sort import SimulatedCamera
from PH.hazards import PERSON_FORWARD_YAW, Apple, Visitor
from PH.manual_control import run_manual_control
from PH.safety import EStop
from PH.scenery_PH import BUCKET_WIDTH, PART_RADIUS, SceneryPH


MANUAL_CONTROL = False
PRE_DROP_WAYPOINT = DO_DIR / "pre_drop_waypoint.json"
# Printed part kind -> (colour bucket, camera image standing in for the photo)
PART_COLOUR = {"body": "brown", "barrel": "black"}
PART_IMAGE = {"body": "BrownPart.jpg", "barrel": "BlackPart.jpg"}
GRASP_FRACTION = 0.5  # TX2-60 grips the printed part at this fraction of its height
DO_SPEEDUP = 10.0  # Do's arm motion is played this many times faster than real time
DO_FRAME_DT = 0.02  # simulated seconds between rendered frames of Do's arm
PERSON_SPEED = 1.2  # walking speed, m/s
PERSON_DT = 0.05  # simulated seconds per walking frame
PERSON_STAND_OFFSET = 0.75  # how far beyond the bucket (+y) the person stands
PERSON_START_DISTANCE = 3.0  # how far back they start walking from
PERSON_PAUSE = 1.5  # seconds spent reaching in for the part
PERSON_CARRY_HEIGHT = 1.0  # height of the carried part above the floor
PARKED_POSITION = (100.0, 100.0, -100.0)  # where finished/removed meshes are hidden


class SortablePart:
    """Present a Do PrintedPart to the TX2-60 as a single pose at its grasp point.

    The printed mesh origin is its footprint corner, so the TX2-60's pick/place pose is
    the footprint centre at mid-height of the part. The part is never teleported.
    """

    keep_pose = True

    def __init__(self, printed_part):
        cfg = PART_CONFIG[printed_part.kind]
        grasp_z = cfg["height"] * GRASP_FRACTION
        self.printed_part = printed_part
        self.kind = printed_part.kind
        self._local = SE3(cfg["width"] / 2, cfg["depth"] / 2, grasp_z)
        # Bucket poses are resting-centre heights of a PART_RADIUS part, so shift for the grasp point
        self.grasp_lift = grasp_z - PART_RADIUS

    @property
    def T(self):
        return (SE3(np.asarray(self.printed_part.mesh.T, dtype=float)) @ self._local).A

    @T.setter
    def T(self, value):
        self.printed_part.mesh.T = (SE3(np.asarray(value, dtype=float)) @ self._local.inv()).A


class _LockedEnv:
    """Env view for the Do thread: same Swift scene, but step() is serialised by a lock."""

    def __init__(self, env, step, lock, gate):
        self._env = env
        self._step = step
        self._lock = lock
        self._gate = gate
        self._pending = 0.0

    def add(self, *args, **kwargs):
        with self._lock:
            return self._env.add(*args, **kwargs)

    def hide(self, obj):
        """Park a mesh far out of sight (Swift's own remove() breaks stepping in this version)."""
        with self._lock:
            parked = np.array(obj.T, dtype=float)
            parked[:3, 3] = PARKED_POSITION
            obj.T = parked

    def real_step(self, dt):
        """Advance the scene in true simulated time (used for the walking person)."""
        self._step(dt)

    def step(self, dt=0.05, *args, **kwargs):
        # Frozen while the E-stop is latched, even for steps that wouldn't render a frame
        self._gate()
        # Shorter sim time per sample, and only render once a display frame's worth has built up:
        # Swift's per-frame overhead otherwise caps how fast the arm can move
        self._pending += dt / DO_SPEEDUP
        if self._pending >= DO_FRAME_DT:
            pending, self._pending = self._pending, 0.0
            self._step(pending, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._env, name)


def _create_do_scenery():
    """Construct Do scenery with scale compatibility for the installed Mesh class."""
    mesh_init = Mesh.__init__

    def mesh_init_with_vector_scale(self, *args, **kwargs):
        scale = kwargs.get("scale")
        if np.isscalar(scale):
            kwargs["scale"] = [scale, scale, scale]
        mesh_init(self, *args, **kwargs)

    Mesh.__init__ = mesh_init_with_vector_scale
    try:
        return SceneryDo()
    finally:
        Mesh.__init__ = mesh_init


def setup_scene(headless=False, headless_cycles=1, safety_gui=None):
    """Create the Swift environment, both scenery sections, and camera callbacks.

    The Do robot runs on a worker thread, so Swift stepping is serialised by a lock. The
    camera and the safety window (Tk) are only ever touched from the main thread: the Do
    thread calls the returned on_part_dropped, which just hands the part over to it.
    safety_gui defaults to on unless headless.
    """
    if safety_gui is None:
        safety_gui = not headless
    env = swift.Swift()
    env.launch(realtime=not headless, headless=headless)

    scenery_ph = SceneryPH()
    scenery_ph.add_to_env(env)
    scenery_do = _create_do_scenery()
    scenery_do.add_to_env(env)

    camera = SimulatedCamera(
        folder=ROOT_DIR / "PH" / "part_images",
        filenames=("BlackPart.jpg", "BrownPart.jpg"),
    )
    swift_step = env.step
    step_lock = threading.RLock()
    dropped = queue.Queue()
    errors = []

    estop = apple = visitor = None
    if safety_gui:
        estop = EStop()
        scenery_ph.tx2_60.safety = estop
        apple = Apple(env, scenery_ph.table_pose, scenery_ph.tx2_60)
        visitor = Visitor(env, scenery_ph.table_pose, estop)
        # A person still in the cell blocks the Reset button
        estop.reset_blocked = lambda: visitor.inside

        def toggle_apple():
            on = apple.toggle()
            apple_button.config(text="Remove obstacle" if on else "Add obstacle")

        apple_button = estop.add_button("Add obstacle", toggle_apple)
        estop.add_button("Person walks in (light curtain)", visitor.trigger)

    def hold_if_stopped():
        """Freeze the whole system while the E-stop is latched.

        The main thread keeps the GUI and the walking person alive (and renders them); the
        Do thread just waits. Nothing else advances, so both arms hold still.
        """
        if estop is None:
            return
        on_main = threading.current_thread() is threading.main_thread()
        while estop.stopped:
            if on_main:
                estop.pump()
                visitor.update(0.05)
                with step_lock:
                    swift_step(0.05)
            else:
                time.sleep(0.05)

    def locked_step(*args, **kwargs):
        hold_if_stopped()
        with step_lock:
            swift_step(*args, **kwargs)
        # Python locks aren't fair; yield so the other robot's thread gets a turn
        time.sleep(0.001)

    def classify_and_queue(part, image_path):
        camera.object_present = True
        try:
            colour = camera.detect_colour(image_path=image_path)
        finally:
            camera.object_present = False
        if colour is None:
            raise RuntimeError("Camera did not classify the dropped part.")
        print(f"General-bin camera detected: {colour}")
        scenery_ph.tx2_60.queue_part(part, colour)

    def step_and_pump_camera(*args, **kwargs):
        locked_step(*args, **kwargs)
        camera.pump()
        if estop is not None:
            estop.pump()
            dt = args[0] if args else kwargs.get("dt", 0.05)
            apple.update(dt)
            visitor.update(dt)
        if errors:
            raise RuntimeError("Do robot sequence failed") from errors[0]
        while not dropped.empty():
            classify_and_queue(*dropped.get())

    env.step = step_and_pump_camera
    env.set_camera_pose([3.0, -3.0, 3.0], [-0.7, 0.85, 0.5])

    def on_part_dropped(part, image_path):
        """Called from the Do thread once its part is in the general bin and its arm is clear."""
        dropped.put((part, image_path))

    sorted_event = threading.Event()
    sorted_count = [0]

    def on_part_sorted(_part, colour):
        print(f"TX2-60 sorted the part into the {colour} bucket")
        camera.close_image()
        sorted_count[0] += 1
        sorted_event.set()
        if headless and sorted_count[0] >= headless_cycles:
            raise SystemExit(0)

    do_env = _LockedEnv(env, locked_step, step_lock, hold_if_stopped)
    return (env, do_env, scenery_ph, scenery_do, on_part_dropped, on_part_sorted,
            sorted_event, errors)


def _load_pre_drop_pose():
    data = json.loads(PRE_DROP_WAYPOINT.read_text(encoding="utf-8"))
    rx, ry, rz = (np.deg2rad(float(data[k])) for k in ("Rx", "Ry", "Rz"))
    pose = (
        SE3(*(float(data[k]) / 1000 for k in ("X", "Y", "Z")))
        @ SE3.Rx(rx) @ SE3.Ry(ry) @ SE3.Rz(rz)
    )
    return pose, data.get("ClearanceByPart", {})


def run_do_sequence(do_env, scenery_ph, scenery_do, on_part_dropped, printer, part):
    """Do's EC66 cycle: take the part from a printer, drop it in the general bin, close up."""
    if part.kind not in PART_COLOUR:
        raise RuntimeError(f"No colour bucket for part kind: {part.kind}")
    T_pre, clearances = _load_pre_drop_pose()
    floor_z = float(scenery_ph.general_bucket_pose.t[2] - PART_RADIUS)
    clearance = (
        float(clearances[part.kind]) / 1000 if part.kind in clearances else None
    )

    tasks = RobotTasks(scenery_do, do_env)
    recorder = tasks.open_door(printer)
    tasks.retract_from_door(printer)
    tasks.move_to_clearance(printer)
    tasks.reorient_for_standby(printer)
    tasks.return_to_standby(printer)
    tasks.move_to_camera_pose(printer)
    tasks.camera_capture_dwell(printer, seconds=3.0)
    tasks.move_to_pre_pick(printer, distance=0.10)
    if not tasks.execute_part_pick(printer, part, tasks.plan_part_pick(printer, part)):
        raise RuntimeError("Pick attachment failed")

    plan = tasks.plan_part_drop(
        part,
        scenery_ph.general_bucket_pose,
        floor_z,
        bucket_width=BUCKET_WIDTH,
        clearance_m=clearance,
        pre_drop_pose=T_pre,
    )
    tasks.execute_part_drop(part, plan, recorder=recorder)
    tasks.return_to_standby(printer)
    # Do's arm is now clear of the bin: the TX2-60 can sort while Do closes the door and homes
    on_part_dropped(SortablePart(part), ROOT_DIR / "PH" / "part_images" / PART_IMAGE[part.kind])
    tasks.return_to_open_handle(printer)
    tasks.close_door(printer)
    tasks.finish_cycle_and_home(printer, recorder=recorder)


def spawn_random_part(do_env, scenery_do):
    """Print a random part in a random printer (the old part and markers are already gone)."""
    # Debug pick markers belong to the original part only
    for marker in scenery_do.pick_markers.values():
        do_env.hide(marker)
    scenery_do.pick_markers.clear()

    printer = random.choice(scenery_do.printers)
    kind = random.choice(list(PART_COLOUR))
    part = PrintedPart(
        plate_reference=printer.bambu_origin,
        kind=kind,
        offset=centered_part_offset(kind),
    )
    scenery_do.parts = {printer: part}
    do_env.add(part.mesh)
    part._env = do_env
    print(f"New {kind} part printed in printer {scenery_do.printers.index(printer) + 1}")
    return printer, part


def person_collects_part(do_env, scenery_ph, part, colour):
    """A person walks to the colour bucket, takes the sorted part and walks off with it.

    Only called once both robots are back in their waiting poses, so the person never
    enters the workspace of a moving arm.
    """
    bucket = scenery_ph.colour_bucket_poses[colour].t
    # The buckets sit on the open (+y) side of the table: approach from there, facing -y
    heading = -np.pi / 2
    stand = np.array([bucket[0], bucket[1] + PERSON_STAND_OFFSET, 0.0])
    start = stand + np.array([PERSON_START_DISTANCE * 0.3, PERSON_START_DISTANCE, 0.0])

    person = getattr(do_env, "_person", None)
    if person is None:
        person = part_mesh("personMaleCasual", pose=SE3(start) @ SE3.Rz(PERSON_FORWARD_YAW))
        person._collision = False
        do_env.add(person)
        do_env._person = person
    print("Person walking to the sorted bucket")

    def place(xy, yaw, carried=False):
        pos = np.array([xy[0], xy[1], 0.0])
        person.T = (SE3(pos) @ SE3.Rz(yaw + PERSON_FORWARD_YAW)).A
        if carried:
            carry_T = np.array(part.mesh.T, dtype=float)
            carry_T[:3, 3] = pos + np.array([0.0, 0.0, PERSON_CARRY_HEIGHT])
            part.mesh.T = carry_T
        do_env.real_step(PERSON_DT)

    def walk(src, dst, carried=False):
        src, dst = np.asarray(src[:2], dtype=float), np.asarray(dst[:2], dtype=float)
        yaw = np.arctan2(*(dst - src)[::-1])
        steps = max(1, int(np.ceil(np.linalg.norm(dst - src) / (PERSON_SPEED * PERSON_DT))))
        for s in range(1, steps + 1):
            place(src + (dst - src) * s / steps, yaw, carried)

    # Walk in, turn to face the bucket, reach in and take the part
    walk(start, stand)
    for _ in range(int(PERSON_PAUSE / PERSON_DT)):
        place(stand[:2], heading)
    print("Person collected the sorted part")
    walk(stand, start, carried=True)
    do_env.hide(part.mesh)
    do_env.hide(person)


def run_do_loop(do_env, scenery_ph, scenery_do, on_part_dropped, sorted_event):
    """Repeat the cycle forever: a person collects each sorted part, then a new one is printed."""
    printer = scenery_do.printer1
    part = scenery_do.parts[printer]
    while True:
        run_do_sequence(do_env, scenery_ph, scenery_do, on_part_dropped, printer, part)
        # Do is home; wait for the TX2-60 to finish sorting and return to its waiting pose
        sorted_event.wait()
        sorted_event.clear()
        person_collects_part(do_env, scenery_ph, part, PART_COLOUR[part.kind])
        printer, part = spawn_random_part(do_env, scenery_do)


def main(headless=False, cycles=1, safety_gui=None):
    """Run Do's pick-and-drop and the TX2-60 sorting together, or the manual-control interface.

    Repeats indefinitely; with headless=True it stops after `cycles` sorted parts.
    """
    (env, do_env, scenery_ph, scenery_do,
     on_part_dropped, on_part_sorted, sorted_event, errors) = setup_scene(headless, cycles, safety_gui)
    tx2_60 = scenery_ph.tx2_60

    if MANUAL_CONTROL:
        scenery_ph.register_obstacles()
        run_manual_control(tx2_60, env)
        return

    def worker():
        try:
            run_do_loop(do_env, scenery_ph, scenery_do, on_part_dropped, sorted_event)
        except BaseException as exc:
            errors.append(exc)

    threading.Thread(target=worker, name="do-robot", daemon=True).start()
    tx2_60.run(
        env,
        scenery_ph.general_bucket_pose,
        scenery_ph.colour_bucket_poses,
        on_sorted=on_part_sorted,
    )
