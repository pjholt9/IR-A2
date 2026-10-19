"""Simulated GUI emergency stop for the TX2-60, separate from the manual joint-control window.

State machine (restart is deliberately a two-step action):
    RUNNING --press E-STOP--> STOPPED --Reset--> RESET_REQUIRED --Restart--> RUNNING
The arm holds its pose while not RUNNING, so restarting resumes the interrupted motion
from exactly where it stopped (no jump, and a carried part stays attached).
"""

import tkinter as tk
from tkinter import ttk

RUNNING = "RUNNING"
STOPPED = "STOPPED"
RESET_REQUIRED = "RESET - ready to restart"

STATE_COLOURS = {RUNNING: "green", STOPPED: "red", RESET_REQUIRED: "orange"}


class EStop:
    """E-stop latch plus its Tk window. Call pump() regularly from the simulation loop."""

    def __init__(self):
        self.state = RUNNING
        self.root = tk.Tk()
        self.root.title("TX2-60 E-STOP")
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)  # closing it must not disarm the stop

        self.status = tk.Label(self.root, text="", font=("Arial", 16, "bold"), fg="white", width=26, pady=8)
        self.status.pack(fill="x")
        self.stop_button = tk.Button(
            self.root, text="E-STOP", font=("Arial", 28, "bold"), bg="red", fg="white",
            activebackground="darkred", height=2, command=self.press,
        )
        self.stop_button.pack(fill="x", padx=10, pady=10)
        row = ttk.Frame(self.root)
        row.pack(fill="x", padx=10, pady=(0, 10))
        self.reset_button = ttk.Button(row, text="1. Reset", command=self.reset)
        self.reset_button.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.restart_button = ttk.Button(row, text="2. Restart", command=self.restart)
        self.restart_button.pack(side="left", expand=True, fill="x", padx=(5, 0))
        self._refresh()

    @property
    def stopped(self):
        return self.state != RUNNING

    def press(self):
        """Latch the stop. Takes effect on the arm's next simulation step."""
        self.state = STOPPED
        self._refresh()

    def reset(self):
        """Clear the stop latch. Only allowed from STOPPED; the arm still does not move."""
        if self.state == STOPPED:
            self.state = RESET_REQUIRED
            self._refresh()

    def restart(self):
        """Resume operation. Only allowed after a reset."""
        if self.state == RESET_REQUIRED:
            self.state = RUNNING
            self._refresh()

    def _refresh(self):
        self.status.config(text=self.state, bg=STATE_COLOURS[self.state])
        self.reset_button.state(["!disabled"] if self.state == STOPPED else ["disabled"])
        self.restart_button.state(["!disabled"] if self.state == RESET_REQUIRED else ["disabled"])

    def pump(self):
        """Process GUI events; call from every simulation step so the buttons stay responsive."""
        try:
            self.root.update()
        except tk.TclError:
            pass
