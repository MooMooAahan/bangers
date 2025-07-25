import tkinter as tk
import os
from PIL import ImageTk, Image
from ui_elements.theme import FG_COLOR_FOR_BUTTON_TEXT, BUTTON_FONT,COLOR_LEFT_AND_RIGHT_INSPECT_BUTTONS,COLOR_LEFT_AND_RIGHT_SAVE_BUTTONS,COLOR_LEFT_AND_RIGHT_SQUISH_BUTTONS
from gameplay.enums import ActionCost

class ButtonMenu(object):
    def __init__(self, root, items):
        self.canvas = tk.Canvas(root, width=500, height=80)
        self.canvas.place(x=35, y=100)
        self.buttons = create_buttons(self.canvas, items)
        create_menu(self.buttons)

    def disable_buttons(self, remaining_time, remaining_humanoids, at_capacity):
        for button in self.buttons:
            button.config(state="normal")
        if remaining_humanoids == 0 or remaining_time <= 0:
            for i in range(0, len(self.buttons)):
                self.buttons[i].config(state="disabled")
        #  Not enough time left? Disable action
        if (remaining_time - ActionCost.SCRAM.value) < ActionCost.SKIP.value:
            self.buttons[0].config(state="disabled")  # Skip
            self.buttons[1].config(state="disabled")  # Inspect (same cost as Skip)
        if (remaining_time - ActionCost.SCRAM.value) < ActionCost.SQUISH.value:
            self.buttons[2].config(state="disabled")  # Squish
        if (remaining_time - ActionCost.SCRAM.value) < ActionCost.SAVE.value:
            self.buttons[3].config(state="disabled")  # Save
        if at_capacity:
            self.buttons[0].config(state="disabled")  # Skip
            self.buttons[1].config(state="disabled")  # Inspect
            self.buttons[2].config(state="disabled")  # Squish
            self.buttons[3].config(state="disabled")  # Save





    def enable_all_buttons(self):
        """Enables button for when you need to retry"""
        for button in self.buttons:
            button.config(state="normal")
            
    
            
            
            
            
            
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
        button.pack(side=tk.TOP, pady=10)


class LeftButtonMenu(object):
    def __init__(self, root, items):
        self.canvas = tk.Canvas(root, width=500, height=80)
        self.canvas.place(x=200, y=160)  # Shifted to x=400 instead of x=100
        self.buttons = self.create_extra_buttons(self.canvas, items)
        create_menu(self.buttons)
    
    def create_extra_buttons(self, canvas, items):
        button_colors= [
        COLOR_LEFT_AND_RIGHT_SQUISH_BUTTONS,
        COLOR_LEFT_AND_RIGHT_SAVE_BUTTONS,
        COLOR_LEFT_AND_RIGHT_INSPECT_BUTTONS
    ]
        buttons = []
        
        for i, item in enumerate(items):
            _, action = item
            color = button_colors[i]
            button = tk.Button(
                canvas,
                text="L",
                height=2,
                width=5,
                font=BUTTON_FONT,
                bg=color,
                fg=FG_COLOR_FOR_BUTTON_TEXT,
                activebackground=color,
                command=action
            )
            buttons.append(button)
        
        return buttons

    def disable_buttons(self, remaining_time, remaining_humanoids, at_capacity):
        for button in self.buttons:
            button.config(state="normal")
        if remaining_humanoids == 0 or remaining_time <= 0:
            for i in range(0, len(self.buttons)):
                self.buttons[i].config(state="disabled")
        #  Not enough time left? Disable action
        if (remaining_time - ActionCost.SCRAM.value) < ActionCost.SKIP.value:
            self.buttons[0].config(state="disabled")  # Inspect
        if (remaining_time - ActionCost.SCRAM.value) < ActionCost.SQUISH.value:
            self.buttons[1].config(state="disabled")  # Squish
        if (remaining_time - ActionCost.SCRAM.value) < ActionCost.SAVE.value:
            self.buttons[2].config(state="disabled")  # Save
        if at_capacity:
            self.buttons[0].config(state="disabled")  # Inspect
            self.buttons[1].config(state="disabled")  # Squish
            self.buttons[2].config(state="disabled")  # Save

    def enable_all_buttons(self):
        """Enables button for when you need to retry"""
        for button in self.buttons:
            button.config(state="normal")
            

class RightButtonMenu(object):
    def __init__(self, root, items):
        self.canvas = tk.Canvas(root, width=500, height=80)
        self.canvas.place(x=270, y=210) 
        self.buttons = self.create_extra_buttons(self.canvas, items)
        create_menu(self.buttons)
    
    def create_extra_buttons(self, canvas, items):
        button_colors =[
        COLOR_LEFT_AND_RIGHT_INSPECT_BUTTONS,
        COLOR_LEFT_AND_RIGHT_SAVE_BUTTONS,
        COLOR_LEFT_AND_RIGHT_SQUISH_BUTTONS
    ]
        buttons = []
        for i, item in enumerate(items):
            _, action = item
            color = button_colors[i]
            button = tk.Button(
                canvas,
                text="R",
                height=2,
                width=5,
                font=BUTTON_FONT,
                bg=color,
                fg=FG_COLOR_FOR_BUTTON_TEXT,
                activebackground=color,
                command=action
            )
            buttons.append(button)
        
        return buttons

    def disable_buttons(self, remaining_time, remaining_humanoids, at_capacity):
        for button in self.buttons:
            button.config(state="normal")
        if remaining_humanoids == 0 or remaining_time <= 0:
            for i in range(0, len(self.buttons)):
                self.buttons[i].config(state="disabled")
        #  Not enough time left? Disable action
        if (remaining_time - ActionCost.SCRAM.value) < ActionCost.SKIP.value:
            self.buttons[0].config(state="disabled")  # Inspect
        if (remaining_time - ActionCost.SCRAM.value) < ActionCost.SQUISH.value:
            self.buttons[1].config(state="disabled")  # Squish
        if (remaining_time - ActionCost.SCRAM.value) < ActionCost.SAVE.value:
            self.buttons[2].config(state="disabled")  # Save
        if at_capacity:
            self.buttons[0].config(state="disabled")  # Inspect
            self.buttons[1].config(state="disabled")  # Squish
            self.buttons[2].config(state="disabled")  # Save

    def enable_all_buttons(self):
        """Enables button for when you need to retry"""
        for button in self.buttons:
            button.config(state="normal")
            
            
            





