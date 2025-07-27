import tkinter as tk
import os
from PIL import ImageTk, Image
from ui_elements.theme import FG_COLOR_FOR_BUTTON_TEXT, BUTTON_FONT,COLOR_LEFT_AND_RIGHT_INSPECT_BUTTONS,COLOR_LEFT_AND_RIGHT_SAVE_BUTTONS,COLOR_LEFT_AND_RIGHT_SQUISH_BUTTONS
from gameplay.enums import ActionCost

class ButtonMenu(object):
    def __init__(self, root, items):
        self.canvas = tk.Canvas(root, width=500, height=80,bg="black",highlightthickness=0)
        self.canvas.place(x=35, y=52)
        self.buttons = create_buttons(self.canvas, items)
        create_menu(self.buttons)
        for i, btn in enumerate(self.buttons):
            btn.update_idletasks()
            print(f"[DEBUG] Main button {i} y={btn.winfo_y()}, height={btn.winfo_height()}")

    def disable_buttons(self, remaining_time, remaining_humanoids, at_capacity):
        for button in self.buttons:
            button.config(state="normal")
        if remaining_humanoids == 0 or remaining_time <= 0:
            for i in range(0, len(self.buttons)):
                self.buttons[i].config(state="disabled")
        #  Not enough time left? Disable action
        # if (remaining_time - ActionCost.SCRAM.value) < ActionCost.SKIP.value:
        #     self.buttons[0].config(state="disabled")  # Skip
        #     self.buttons[1].config(state="disabled")  # Inspect (same cost as Skip)
        # if (remaining_time - ActionCost.SCRAM.value) < ActionCost.SQUISH.value:
        #     self.buttons[2].config(state="disabled")  # Squish
        # if (remaining_time - ActionCost.SCRAM.value) < ActionCost.SAVE.value:
        #     self.buttons[3].config(state="disabled")  # Save





    def enable_all_buttons(self):
        """Enables button for when you need to retry"""
        for button in self.buttons:
            button.config(state="normal")
            
    def force_disable_all_buttons(self):
        """Force disable all buttons regardless of conditions (for game end)"""
        for button in self.buttons:
            button.config(state="disabled")

def create_buttons(canvas, items):
    buttons = []
    for item in items:
        if len(item) == 3:
            text, action, color = item
        else:
            text, action = item
            color = "#cccccc"  # fallback

        buttons.append(tk.Button(
            canvas,
            text=text,
            height=2,
            width=15,
            command=action,
            font=BUTTON_FONT,
            bg=color,
            fg=FG_COLOR_FOR_BUTTON_TEXT,
            activebackground=color
        ))
    return buttons


def create_menu(buttons):
    for button in buttons:
        button.pack(side=tk.TOP, pady=20)


class LeftButtonMenu(object):
    def __init__(self, root, items):
        self.buttons = []
        x = 35           # Horizontal position of "L" buttons
        y_start = 215       # Matches y-position of "Skip" button
        y_gap = 92
        
        button_colors = [
            COLOR_LEFT_AND_RIGHT_INSPECT_BUTTONS,
            COLOR_LEFT_AND_RIGHT_SAVE_BUTTONS,
            COLOR_LEFT_AND_RIGHT_SQUISH_BUTTONS
        ]

        for i, item in enumerate(items):
            _, action = item
            color = button_colors[i]
            button = tk.Button(
                root,
                text="L",
                height=1,
                width=7,
                font=BUTTON_FONT,
                bg=color,
                fg=FG_COLOR_FOR_BUTTON_TEXT,
                activebackground=color,
                relief="raised",
                bd=2,
                command=action
            )
            button.place(x=x, y=y_start + y_gap * i)
            self.buttons.append(button)

    def disable_buttons(self, remaining_time, remaining_humanoids, at_capacity):
        for button in self.buttons:
            button.config(state="normal")
        if remaining_humanoids == 0 or remaining_time <= 0:
        # Start by enabling all
          for button in self.buttons:
              button.config(state="normal")

        # Disable based on time
        if at_capacity:
            self.buttons[2].config(state="disabled")

    def enable_all_buttons(self):
        for button in self.buttons:
            button.config(state="normal")
            
    def force_disable_all_buttons(self):
        """Force disable all buttons regardless of conditions (for game end)"""
        for button in self.buttons:
            button.config(state="disabled")
            

class RightButtonMenu(object):
    def __init__(self, root, items):
        self.buttons = []
        x = 115            # Adjust horizontally to right side of main buttons
        y_start = 215      # Match LeftButtonMenu start (below Inspect)
        y_gap = 92         # Match spacing of main buttons

        button_colors = [
            COLOR_LEFT_AND_RIGHT_INSPECT_BUTTONS,
            COLOR_LEFT_AND_RIGHT_SAVE_BUTTONS,
            COLOR_LEFT_AND_RIGHT_SQUISH_BUTTONS
        ]

        for i, item in enumerate(items):
            _, action = item
            color = button_colors[i]
            button = tk.Button(
                root,
                text="R",
                height=1,
                width=7,
                font=BUTTON_FONT,
                bg=color,
                fg=FG_COLOR_FOR_BUTTON_TEXT,
                activebackground=color,
                relief="raised",
                bd=2,
                command=action
            )
            button.place(x=x, y=y_start + i * y_gap)
            self.buttons.append(button)

    def disable_buttons(self, remaining_time, remaining_humanoids, at_capacity):
        for button in self.buttons:
            button.config(state="normal")
        if remaining_humanoids == 0 or remaining_time <= 0:
            for i in range(0, len(self.buttons)):
                self.buttons[i].config(state="disabled")
        #  Not enough time left? Disable action
        # if (remaining_time - ActionCost.SCRAM.value) < ActionCost.SKIP.value:
        #     self.buttons[0].config(state="disabled")  # Inspect
        # if (remaining_time - ActionCost.SCRAM.value) < ActionCost.SQUISH.value:
        #     self.buttons[1].config(state="disabled")  # Squish
        # if (remaining_time - ActionCost.SCRAM.value) < ActionCost.SAVE.value:
        #     self.buttons[2].config(state="disabled")  # Save
        if at_capacity:
            self.buttons[2].config(state="disabled")  # Save

    def enable_all_buttons(self):
        """Enables button for when you need to retry"""
        for button in self.buttons:
            button.config(state="normal")
            
    def force_disable_all_buttons(self):
        """Force disable all buttons regardless of conditions (for game end)"""
        for button in self.buttons:
            button.config(state="disabled")
            
            
            





