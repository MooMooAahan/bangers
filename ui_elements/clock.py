import tkinter as tk
import math

class Clock(object):
    def __init__(self, root, w, h, init_h, init_m):
        # Smaller canvas
        self.canvas = tk.Canvas(root, width=180, height=120, bg="#f0f0f0", highlightthickness=0)
        self.canvas.place(x=math.floor(0.80 * w), y=30)

        # Title label
        tk.Label(self.canvas, text="Current time", font=("Arial", 15), bg="#f0f0f0").place(relx=0.5, y=10, anchor="n")

        # Digital time label (smaller)
        self.label = tk.Label(self.canvas, text="", font=("Arial", 30, "bold"), bg="#f0f0f0", fg="midnightblue")
        self.label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.blink = True
        self.current_h = init_h
        self.current_m = init_m
        self._start_blink()

    def update_time(self, h, m):
        self.current_h = h
        self.current_m = m
        self._render_time()

    def _render_time(self):
        hour_24 = self.current_h
        minute = self.current_m
        am_pm = "AM" if hour_24 < 12 else "PM"
        hour_12 = hour_24 % 12
        if hour_12 == 0:
            hour_12 = 12
        colon = ":" if self.blink else " "
        time_str = f"{hour_12}{colon}{minute:02d} {am_pm}"
        self.label.config(text=time_str)

    def _start_blink(self):
        self.blink = not self.blink
        self._render_time()
        self.label.after(500, self._start_blink)
