import tkinter as tk
import os
from PIL import ImageTk, Image

from gameplay.enums import ActionCost

class ButtonMenu(object):
    def __init__(self, root, items):
        self.canvas = tk.Canvas(root, width=500, height=80)
        self.canvas.place(x=100, y=150)
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
        (text, action) = item
        buttons.append(tk.Button(canvas, text=text, height=2, width=15,
                                 command=action))
    return buttons


def create_menu(buttons):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graphics', 'logo.png')
    logo = ImageTk.PhotoImage(Image.open(path).resize((300, 50), Image.LANCZOS))
    label = tk.Label(image=logo)
    label.image = logo

    # Position image
    label.place(x=10, y=10)

    for button in buttons:
        button.pack(side=tk.TOP, pady=10)


class LeftButtonMenu(object):
    def __init__(self, root, items):
        self.canvas = tk.Canvas(root, width=500, height=80)
        self.canvas.place(x=220, y=210)  # Shifted to x=400 instead of x=100
        self.buttons = self.create_extra_buttons(self.canvas, items)
        create_menu(self.buttons)
    
    def create_extra_buttons(self, canvas, items):
        buttons = []
        for item in items:
            (text, action) = item
            buttons.append(tk.Button(canvas, text="L", height=2, width=5,
                                     command=action))
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
        self.canvas.place(x=270, y=210)  # Shifted to x=400 instead of x=100
        self.buttons = self.create_extra_buttons(self.canvas, items)
        create_menu(self.buttons)
    
    def create_extra_buttons(self, canvas, items):
        buttons = []
        for item in items:
            (text, action) = item
            buttons.append(tk.Button(canvas, text="R", height=2, width=5,
                                     command=action))
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
            
            
            





