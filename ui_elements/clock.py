from ui_elements.theme import BG_COLOR_FOR_CLOCK_TITLE, FG_COLOR_FOR_TIME,FG_COLOR_FOR_CURRENT_TIME,TITLE_FONT_FOR_CURRENT_TIME,BG_COLOR_FOR_CLOCK
import tkinter as tk
from tkinter import font
import math

class Clock(object):
    def __init__(self, root, w, h, init_h, init_m):
        self.canvas = tk.Canvas(root, width=180, height=100, bg = "#5B7B7A", highlightthickness=2, relief="groove", bd=2)
        # Place the clock at the top center
        self.canvas.place(x=1060, y=70)

        time_font = font.Font(family="Fixedsys", size=30, weight="bold")
        title_font = font.Font(family="Fixedsys", size=14)

        # Title label
        tk.Label(self.canvas, text="Current time", font=title_font, bg=BG_COLOR_FOR_CLOCK_TITLE,fg=FG_COLOR_FOR_CURRENT_TIME).place(relx=0.5, y=10, anchor="n")

        # Digital time label

        self.label = tk.Label(self.canvas, text="", font=time_font, bg="#5B7B7A", fg= "#FFFFFF")
        self.label.place(relx=0.5, rely=0.65, anchor=tk.CENTER)

        self.blink = True
        self.current_h = init_h
        self.current_m = init_m
        self._start_blink()

    def update_time(self, h, m, force_pm=False):
        self.current_h = h
        self.current_m = m
        self.force_pm = force_pm
        self._render_time()

    def _render_time(self):
        hour_24 = self.current_h
        minute = self.current_m
        # Use force_pm if set, otherwise normal AM/PM logic
        if hasattr(self, 'force_pm') and self.force_pm:
            am_pm = "PM"
        else:
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
