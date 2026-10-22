"""Assemble and run the combined PH and Do workcell."""

import json
import queue
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
# Do's files import each other without a package prefix, so his folder has to be on the path
sys.path.insert(0, str(DO_DIR))

# Do's side of the cell: scenery (printers, rail, EC66), printed parts and his task sequence
from Do.scenery import Scenery as SceneryDo
from printed_part import PART_CONFIG, PrintedPart, centered_part_offset
from robot_tasks import RobotTasks

# My side of the cell: camera, hazards (apple/person), e-stop, manual GUI and TX2-60 scenery
from PH.colour_sort import SimulatedCamera
from PH.hazards import PERSON_FORWARD_YAW, Apple, Visitor
from PH.manual_control import run_manual_control
from PH.manual_panel import ManualPanel
from PH.safety import EStop
from PH.scenery_PH import BUCKET_WIDTH, PART_RADIUS, SceneryPH


# set True to skip the cycle and just open the old standalone joint-slider window
MANUAL_CONTROL = False
# the pre-drop pose Do taught for his arm (where it holds the part over the general bin)
PRE_DROP_WAYPOINT = DO_DIR / "pre_drop_waypoint.json"
# Printed part kind -> (colour bucket, camera image standing in for the photo)
PART_COLOUR = {"body": "brown", "barrel": "black"}
PART_IMAGE = {"body": "BrownPart.jpg", "barrel": "BlackPart.jpg"}
GRASP_FRACTION = 0.5  # TX2-60 grips the printed part at this fraction of its height
DO_SPEEDUP = 5.0  # Do's arm motion is played this many times faster than real time
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

    # tells tx2_60.sort_part not to teleport the part to the bin, it's already there
    keep_pose = True

    def __init__(self, printed_part):
        # size of this kind of part (body/barrel) from Do's config
        cfg = PART_CONFIG[printed_part.kind]
        grasp_z = cfg["height"] * GRASP_FRACTION
        self.printed_part = printed_part
        self.kind = printed_part.kind
        # offset from the mesh origin (a corner) to the point I actually grip: footprint centre, mid height
        self._local = SE3(cfg["width"] / 2, cfg["depth"] / 2, grasp_z)
        # Bucket poses are resting-centre heights of a PART_RADIUS part, so shift for the grasp point
        self.grasp_lift = grasp_z - PART_RADIUS

    # reading T gives the grasp point in world coordinates (mesh pose * local offset)
    @property
    def T(self):
        return (SE3(np.asarray(self.printed_part.mesh.T, dtype=float)) @ self._local).A

    # writing T (the gripper dragging the part) moves the mesh so the grasp point lands there
    @T.setter
    def T(self, value):
        self.printed_part.mesh.T = (SE3(np.asarray(value, dtype=float)) @ self._local.inv()).A


class _LockedEnv:
    """Env view for the Do thread: same Swift scene, but step() is serialised by a lock."""

    def __init__(self, env, step, lock, gate):
        self._env = env
        self._step = step   # the real (locked) Swift step
        self._lock = lock   # shared with the main thread so we never step Swift at the same time
        self._gate = gate   # blocks while the e-stop is latched
        self._pending = 0.0  # simulated time waiting to be rendered (see step)

    # adding meshes also needs the lock, otherwise Swift can be mid-step on the other thread
    def add(self, *args, **kwargs):
        with self._lock:
            return self._env.add(*args, **kwargs)

    def hide(self, obj):
        """Park a mesh far out of sight (Swift's own remove() breaks stepping in this version)."""
        with self._lock:
            # keep the rotation, just move the translation way off the map
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
        # enough time has built up for a frame: render it all in one step and reset the counter
        if self._pending >= DO_FRAME_DT:
            pending, self._pending = self._pending, 0.0
            self._step(pending, *args, **kwargs)

    def step_with_pose(self, robot, q, door, angle, dt=0.05):
        """Render a door/robot pair atomically, every opening/closing sample.

        No frame skipping for synchronized contact motion. Respect E-stop first.
        """
        self._gate()
        with self._lock:
            # move the door and the arm together so the handle never visibly lags the gripper
            door.update_angle(float(angle))
            robot.q = np.asarray(q, dtype=float).copy()
            # Flush accumulated time from ordinary motions before the contact frame.
            frame_dt = self._pending + dt / DO_SPEEDUP
            self._pending = 0.0
            # Call the original locked Swift step, NOT env.step (which also pumps Tk).
            # env.step is wrapped by step_and_pump_camera and is main-thread-only.
            self._step(frame_dt)
        time.sleep(0.001)

    # anything not overridden here (set_camera_pose etc.) just falls through to the real env
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


def setup_scene(headless=False, headless_cycles=1, safety_gui=None, controller=None):
    """Create the Swift environment, both scenery sections, and camera callbacks.

    The Do robot runs on a worker thread, so Swift stepping is serialised by a lock. The
    camera and the safety window (Tk) are only ever touched from the main thread: the Do
    thread calls the returned on_part_dropped, which just hands the part over to it.
    safety_gui defaults to on unless headless.
    """
    # launch Swift (realtime browser window unless we're running headless for testing)
    if safety_gui is None:
        safety_gui = not headless
    env = swift.Swift()
    env.launch(realtime=not headless, headless=headless)

    # build both halves of the cell into the same Swift scene
    scenery_ph = SceneryPH()
    scenery_ph.add_to_env(env)
    scenery_do = _create_do_scenery()
    scenery_do.add_to_env(env)

    # fake camera over the general bin, it shows a photo and "detects" the colour from it
    camera = SimulatedCamera(
        folder=ROOT_DIR / "PH" / "part_images",
        filenames=("BlackPart.jpg", "BrownPart.jpg"),
    )
    swift_step = env.step  # keep the original step, env.step gets replaced further down
    step_lock = threading.RLock()  # only one thread may step Swift at a time
    dropped = queue.Queue()  # parts Do has dropped, waiting for the main thread to classify them
    errors = []  # exceptions from Do's thread, re-raised on the main thread

    # all the safety stuff is optional (off when running headless)
    estop = apple = visitor = None
    if safety_gui:
        estop = EStop()
        scenery_ph.tx2_60.safety = estop  # my arm checks this every step and holds when it's latched
        apple = Apple(env, scenery_ph.table_pose, scenery_ph.tx2_60)
        visitor = Visitor(env, scenery_ph.table_pose, estop)
        # only a person inside the cell blocks the Reset button (the apple isn't a stop)
        estop.reset_blocked = lambda: visitor.inside

        # the apple is just an obstacle: it doesn't stop anything, my arm waits for it to
        # move out of the way (collision check in tx2_60) and carries on
        def toggle_apple():
            on = apple.toggle()
            apple_button.config(text="Remove obstacle" if on else "Add obstacle")

        apple_button = estop.add_button("Add obstacle", toggle_apple)
        estop.add_button("Person Intrusion (Light Curtain)", visitor.trigger)
        # production buttons + manual panel get added to the same window
        if controller is not None:
            controller.install_gui(estop, scenery_do, scenery_ph.tx2_60, env,
                                   scenery_ph.general_bucket_pose)

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
                # keep the GUI responsive, let the person keep walking, and keep rendering
                estop.pump()
                visitor.update(0.05)
                with step_lock:
                    swift_step(0.05)
            else:
                time.sleep(0.05)

    # every Swift step goes through here: wait if stopped, then step under the lock
    def locked_step(*args, **kwargs):
        hold_if_stopped()
        with step_lock:
            swift_step(*args, **kwargs)
        # Python locks aren't fair; yield so the other robot's thread gets a turn
        time.sleep(0.001)

    # show the part to the camera, read its colour, then hand it to my arm's sorting queue
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

    # replaces env.step so that any step my arm takes also updates the GUI, camera and hazards
    def step_and_pump_camera(*args, **kwargs):
        # The Swift scene is shared with a worker thread; Tk must never be pumped there.
        locked_step(*args, **kwargs)
        if threading.current_thread() is not threading.main_thread():
            return
        camera.pump()
        if controller is not None:
            controller.pump()
        if estop is not None:
            estop.pump()
            # move the apple and the visitor forward by however long this step was
            dt = args[0] if args else kwargs.get("dt", 0.05)
            apple.update(dt)
            visitor.update(dt)
        # if Do's thread crashed, fail loudly here instead of silently stalling
        if errors:
            raise RuntimeError("Do robot sequence failed") from errors[0]
        # classify anything Do has dropped since last step (camera has to run on the main thread)
        while not dropped.empty():
            classify_and_queue(*dropped.get())

    env.step = step_and_pump_camera
    # camera looking at the cell from the corner, aimed at roughly the middle of the table
    env.set_camera_pose([3.0, -3.0, 3.0], [-0.7, 0.85, 0.5])

    def on_part_dropped(part, image_path):
        """Called from the Do thread once its part is in the general bin and its arm is clear."""
        dropped.put((part, image_path))

    # set once my arm has finished a sort and is back at standby (Do's thread waits on this)
    sorted_event = threading.Event()
    sorted_count = [0]

    def on_part_sorted(_part, colour):
        print(f"TX2-60 sorted the part into the {colour} bucket")
        camera.close_image()
        sorted_count[0] += 1
        sorted_event.set()
        # headless test mode: stop after the requested number of sorted parts
        if headless and sorted_count[0] >= headless_cycles:
            raise SystemExit(0)

    do_env = _LockedEnv(env, locked_step, step_lock, hold_if_stopped)
    return (env, do_env, scenery_ph, scenery_do, on_part_dropped, on_part_sorted,
            sorted_event, errors)


# read Do's taught pre-drop pose (mm and degrees in the json) as an SE3 in metres
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
    # height of the general bucket floor, so Do knows how far down to lower the part
    floor_z = float(scenery_ph.general_bucket_pose.t[2] - PART_RADIUS)
    clearance = (
        float(clearances[part.kind]) / 1000 if part.kind in clearances else None
    )

    # Do's steps in order: open the door, back out, camera look, pick the part
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

    # Do drops the part, then goes back to standby via a clearance pose so he doesn't hit the bin
    plan = tasks.plan_part_drop(
        part,
        scenery_ph.general_bucket_pose,
        floor_z,
        bucket_width=BUCKET_WIDTH,
        clearance_m=clearance,
        pre_drop_pose=T_pre,
    )
    tasks.execute_part_drop(part, plan, recorder=recorder)
    tasks.return_to_standby_via_clearance(
        printer, DO_DIR / "post_drop_clearance.json", recorder=recorder
    )
    # Do's arm is now clear of the bin: the TX2-60 can sort while Do closes the door and homes
    on_part_dropped(SortablePart(part), ROOT_DIR / "PH" / "part_images" / PART_IMAGE[part.kind])
    tasks.return_to_open_handle(printer)
    tasks.close_door(printer)
    tasks.finish_cycle_and_home(printer, recorder=recorder)


# drop a new part onto a printer's build plate (only when the operator asks for one)
def spawn_selected_part(do_env, scenery_do, printer, kind):
    """Spawn only on operator request, at the calibrated Bambu Studio origin."""
    if kind not in PART_COLOUR:
        raise ValueError(f"Unknown part type: {kind}")
    # The printer must have no existing queued or in-progress part.
    # (the little green marker from Do's scenery gets parked out of sight)
    old_marker = scenery_do.pick_markers.pop(printer, None)
    if old_marker is not None:
        do_env.hide(old_marker)
    part = PrintedPart(
        plate_reference=printer.bambu_origin,
        kind=kind,
        offset=centered_part_offset(kind),
    )
    scenery_do.parts[printer] = part
    do_env.add(part.mesh)
    part._env = do_env  # Do's part code steps this env when it's carried
    print(f"Spawned {kind} in printer {scenery_do.printers.index(printer) + 1}")
    return part


class ProductionController:
    """Thread-safe operator commands; Tk callbacks never execute robot motions."""

    def __init__(self):
        # GUI -> worker: spawn/start requests. worker -> GUI: status text and button state
        self.commands = queue.Queue()
        self.updates = queue.Queue()
        self.pending = {}  # printer index -> PrintedPart, owned by worker
        self.busy = False  # true while a cycle is running
        self.estop = None
        self.status = None
        self.printer_labels = {}
        self.start_button = None
        self.panel = None  # manual control panel, created in install_gui

    # true while the manual window is open (or the arm is still walking back to standby)
    @property
    def manual(self):
        return self.panel is not None and self.panel.active

    # only allow manual mode when nothing is running, so the two modes can't fight over the arm
    def request_manual(self, tx2_60):
        if self.panel.active:
            self.panel.open()
        elif self.busy or tx2_60.busy or tx2_60.pending_parts or self.commands.qsize():
            self.status.config(text="Manual mode needs an idle cell: wait for the cycle to finish")
        elif self.estop.stopped:
            self.status.config(text="Reset + Restart safety before entering manual mode")
        else:
            self.panel.open()
            self.status.config(text="MANUAL MODE: spawn/start disabled until you exit manual mode")
            self.start_button.state(["disabled"])

    # called by the panel once the arm is back at standby
    def _manual_closed(self):
        self.status.config(text="Manual mode ended: arm back at standby")
        self.start_button.state(["disabled"] if self.busy else ["!disabled"])

    # build the production controls into the e-stop window (Tk, main thread only)
    def install_gui(self, estop, scenery_do, tx2_60=None, env=None, general_bucket_pose=None):
        import tkinter as tk
        from tkinter import ttk
        self.estop = estop
        root = estop.root
        root.title("Industrial Robotics | EC66 + TX2-60")
        root.geometry("430x690")
        estop.status.config(font=("Arial", 13, "bold"))
        frame = ttk.LabelFrame(root, text="Production Control")
        # Existing E-stop and incident buttons stay visible and functional.
        frame.pack(fill="x", padx=10, pady=(5, 8), before=estop.stop_button)
        ttk.Label(frame, text="AUTOMATIC MODE", font=("Arial", 12, "bold")).pack(pady=5)
        # manual joint/XYZ/gripper window (opens when the button below is pressed)
        self.panel = ManualPanel(root, tx2_60, env, general_bucket_pose, on_close=self._manual_closed)
        ttk.Button(frame, text="MANUAL: TX2-60 joints + gripper",
                   command=lambda: self.request_manual(tx2_60)).pack(fill="x", padx=8, pady=4)
        # one box per printer with spawn buttons for each part type
        for i in (0, 1):
            sub = ttk.LabelFrame(frame, text=f"Printer {i + 1}")
            sub.pack(fill="x", padx=8, pady=4)
            row = ttk.Frame(sub)
            row.pack(fill="x", padx=5, pady=4)
            ttk.Button(row, text="Spawn Barrel", command=lambda n=i: self.request_spawn(n, "barrel")).pack(side="left", expand=True, fill="x", padx=2)
            ttk.Button(row, text="Spawn Body", command=lambda n=i: self.request_spawn(n, "body")).pack(side="left", expand=True, fill="x", padx=2)
            lbl = ttk.Label(sub, text="Empty")
            lbl.pack(pady=(0, 3))
            self.printer_labels[i] = lbl
        self.start_button = ttk.Button(frame, text="START ONE CYCLE", command=self.request_start)
        self.start_button.pack(fill="x", padx=8, pady=7)
        self.status = ttk.Label(frame, text="READY — spawn a part to begin", wraplength=380)
        self.status.pack(padx=8, pady=(0, 8))

    # the GUI never moves robots itself, it just queues a command for Do's worker thread
    def request_spawn(self, index, kind):
        if self.estop.stopped or self.busy or self.manual:
            self.status.config(text="Cannot spawn while stopped or while a cycle is running")
            return
        self.commands.put(("spawn", index, kind))
        self.status.config(text=f"Requested {kind} in Printer {index + 1}...")

    # same idea for start: refuse if stopped, busy or in manual mode
    def request_start(self):
        if self.estop.stopped or self.busy or self.manual:
            self.status.config(text="Reset + Restart safety before starting; or wait for cycle")
            return
        self.commands.put(("start",))
        self.status.config(text="Start requested...")

    def pump(self):
        # Called only from the main simulation thread.
        if self.panel is not None:
            self.panel.pump()
        # apply any status/printer/busy updates the worker thread has queued up
        while True:
            try:
                msg = self.updates.get_nowait()
            except queue.Empty:
                break
            tag = msg[0]
            if tag == "status":
                self.status.config(text=msg[1])
            elif tag == "printer":
                self.printer_labels[msg[1]].config(text=msg[2])
            elif tag == "busy":
                self.busy = msg[1]
                self.start_button.state(["disabled"] if self.busy else ["!disabled"])



def person_collects_part(do_env, scenery_ph, part, colour):
    """A person walks to the colour bucket, takes the sorted part and walks off with it.

    Only called once both robots are back in their waiting poses, so the person never
    enters the workspace of a moving arm.
    """
    # work out the walking route: stand just past the bucket, start further back and off to the side
    bucket = scenery_ph.colour_bucket_poses[colour].t
    # The buckets sit on the open (+y) side of the table: approach from there, facing -y
    heading = -np.pi / 2
    stand = np.array([bucket[0], bucket[1] + PERSON_STAND_OFFSET, 0.0])
    start = stand + np.array([PERSON_START_DISTANCE * 0.3, PERSON_START_DISTANCE, 0.0])

    # make the person mesh once and reuse it every cycle (adding/removing meshes is flaky in Swift)
    person = getattr(do_env, "_person", None)
    if person is None:
        person = part_mesh("personMaleCasual", pose=SE3(start) @ SE3.Rz(PERSON_FORWARD_YAW))
        person._collision = False
        do_env.add(person)
        do_env._person = person
    print("Person walking to the sorted bucket")

    # put the person at xy facing yaw (and move the part with them if they're carrying it), then render a frame
    def place(xy, yaw, carried=False):
        pos = np.array([xy[0], xy[1], 0.0])
        person.T = (SE3(pos) @ SE3.Rz(yaw + PERSON_FORWARD_YAW)).A
        if carried:
            carry_T = np.array(part.mesh.T, dtype=float)
            carry_T[:3, 3] = pos + np.array([0.0, 0.0, PERSON_CARRY_HEIGHT])
            part.mesh.T = carry_T
        do_env.real_step(PERSON_DT)

    # straight-line walk from src to dst in small steps at PERSON_SPEED, facing the direction of travel
    def walk(src, dst, carried=False):
        src, dst = np.asarray(src[:2], dtype=float), np.asarray(dst[:2], dtype=float)
        yaw = np.arctan2(*(dst - src)[::-1])  # arctan2(dy, dx)
        steps = max(1, int(np.ceil(np.linalg.norm(dst - src) / (PERSON_SPEED * PERSON_DT))))
        for s in range(1, steps + 1):
            place(src + (dst - src) * s / steps, yaw, carried)

    # Walk in, turn to face the bucket, reach in and take the part
    walk(start, stand)
    # stand still (re-render the same pose) for PERSON_PAUSE seconds while "reaching in"
    for _ in range(int(PERSON_PAUSE / PERSON_DT)):
        place(stand[:2], heading)
    print("Person collected the sorted part")
    walk(stand, start, carried=True)
    # they're gone with the part: park both out of sight until next time
    do_env.hide(part.mesh)
    do_env.hide(person)


# Do's thread: waits for operator commands from the GUI and runs the robots for each one
def run_operator_loop(controller, do_env, scenery_ph, scenery_do, on_part_dropped, sorted_event):
    """One operator-selected cycle per START; no automatic random spawning."""
    # Remove the default demonstration part so printers begin empty.
    for printer, part in list(scenery_do.parts.items()):
        do_env.hide(part.mesh)
        scenery_do.parts.pop(printer, None)
    for printer, marker in list(scenery_do.pick_markers.items()):
        do_env.hide(marker)
        scenery_do.pick_markers.pop(printer, None)
    for index in (0, 1):
        controller.updates.put(("printer", index, "Empty"))
    controller.updates.put(("status", "READY — select a printer and spawn a part"))

    while True:
        # block here until the GUI sends a command
        command = controller.commands.get()
        action = command[0]
        if action == "spawn":
            _, index, kind = command
            # ignore spawns that would clash with a running cycle, manual mode or an e-stop
            if controller.busy or controller.manual or controller.estop.stopped:
                controller.updates.put(("status", "Spawn refused: cell busy or safety stop active"))
                continue
            if index in controller.pending:
                controller.updates.put(("status", f"Printer {index + 1} already contains a part"))
                continue
            printer = scenery_do.printers[index]
            part = spawn_selected_part(do_env, scenery_do, printer, kind)
            controller.pending[index] = part
            controller.updates.put(("printer", index, f"Ready: {kind}"))
            controller.updates.put(("status", "Part ready — press START ONE CYCLE"))
        elif action == "start":
            if controller.busy or controller.manual or controller.estop.stopped:
                controller.updates.put(("status", "Start refused: busy or safety stop active"))
                continue
            if not controller.pending:
                controller.updates.put(("status", "Spawn a part before START"))
                continue
            index = next(iter(controller.pending))  # take the first printer that has a part waiting
            part = controller.pending.pop(index)
            printer = scenery_do.printers[index]
            controller.busy = True
            controller.updates.put(("busy", True))
            controller.updates.put(("printer", index, f"Processing: {part.kind}"))
            controller.updates.put(("status", f"RUNNING — Printer {index + 1} / {part.kind}"))
            sorted_event.clear()
            # Do's whole pick-and-drop runs here (blocks until his arm has finished)
            run_do_sequence(do_env, scenery_ph, scenery_do, on_part_dropped, printer, part)
            # The sorting callback fires once TX2-60 returns to its waiting pose.
            sorted_event.wait()
            # Preserve Patrick's collection animation, but don't auto-spawn another part.
            person_collects_part(do_env, scenery_ph, part, PART_COLOUR[part.kind])
            scenery_do.parts.pop(printer, None)
            controller.updates.put(("printer", index, "Empty"))
            controller.busy = False
            controller.updates.put(("busy", False))
            controller.updates.put(("status", "CYCLE COMPLETE — waiting for operator"))



def run_headless_loop(do_env, scenery_ph, scenery_do, on_part_dropped, sorted_event):
    """Legacy headless regression mode (GUI operator mode does not use this)."""
    import random
    # start with the demo part already sitting in printer 1, then keep picking random ones
    printer = scenery_do.printer1
    part = scenery_do.parts[printer]
    while True:
        sorted_event.clear()
        run_do_sequence(do_env, scenery_ph, scenery_do, on_part_dropped, printer, part)
        sorted_event.wait()
        person_collects_part(do_env, scenery_ph, part, PART_COLOUR[part.kind])
        printer = random.choice(scenery_do.printers)
        part = spawn_selected_part(do_env, scenery_do, printer, random.choice(list(PART_COLOUR)))


def main(headless=False, cycles=1, safety_gui=None):
    """Run Do's pick-and-drop and the TX2-60 sorting together, or the manual-control interface.

    Repeats indefinitely; with headless=True it stops after `cycles` sorted parts.
    """
    # the GUI only exists in the real (non-headless) run
    controller = ProductionController() if not headless and safety_gui is not False else None
    (env, do_env, scenery_ph, scenery_do,
     on_part_dropped, on_part_sorted, sorted_event, errors) = setup_scene(
        headless, cycles, safety_gui, controller=controller)
    tx2_60 = scenery_ph.tx2_60

    if MANUAL_CONTROL:
        scenery_ph.register_obstacles()
        run_manual_control(tx2_60, env)
        return

    # Do's robot runs on its own thread so it and my arm move at the same time
    def worker():
        try:
            if controller is not None:
                run_operator_loop(controller, do_env, scenery_ph, scenery_do, on_part_dropped, sorted_event)
            else:
                # Headless regression mode retains the original automated sequence.
                run_headless_loop(do_env, scenery_ph, scenery_do, on_part_dropped, sorted_event)
        except BaseException as exc:
            errors.append(exc)

    threading.Thread(target=worker, name="do-robot", daemon=True).start()
    # my arm's loop stays on the main thread: waits for queued parts, sorts them, returns to standby
    tx2_60.run(
        env,
        scenery_ph.general_bucket_pose,
        scenery_ph.colour_bucket_poses,
        on_sorted=on_part_sorted,
    )
