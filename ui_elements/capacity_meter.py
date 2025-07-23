import math
import tkinter as tk
from PIL import Image, ImageTk
import os
import tkinter.messagebox  # <-- Add this import


class CapacityMeter(object):
    def __init__(self, root, w, h, max_cap, get_ambulance_contents=None):
        self.canvas = tk.Canvas(root, width=math.floor(0.2 * w), height=math.floor(0.5 * h))
        # Align with the clock (placed at x=math.floor(0.75 * w))
        x_pos = math.floor(0.75 * w)
        y_pos = math.floor(0.4 * h) - 20  # move up by 20px
        self.canvas.place(x=x_pos, y=y_pos)
        self.__units = []
        self.unit_size = 14  # Half the previous size (was 40)
        self.canvas.update()
        # Robust image path
        image_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'ambulance.png')
        self.bg_image = Image.open(image_path)
        canvas_width = int(self.canvas.winfo_width())
        canvas_height = int(self.canvas.winfo_height())
        # Scale image by 1.5x but keep aspect ratio
        max_width = int(canvas_width * 1.5)
        max_height = int(canvas_height * 1.5)
        img_w, img_h = self.bg_image.size
        scale = min(max_width / img_w, max_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        self.bg_image = self.bg_image.resize((new_w, new_h), Image.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)
        # Center the image in the canvas, with additional offset and shift up 30px
        x_offset = (canvas_width - new_w) // 2 + 25
        y_offset = (canvas_height - new_h) // 2 + 30 - 30
        self.bg_image_id = self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.bg_photo)
        self.render(max_cap, self.unit_size)

        # Store the callback or data provider for ambulance contents
        self.get_ambulance_contents = get_ambulance_contents
        # Bind click event to ambulance image
        self.canvas.tag_bind(self.bg_image_id, '<Button-1>', self.on_ambulance_click)

    def on_ambulance_click(self, event):
        # Get ambulance contents from callback or attribute
        if self.get_ambulance_contents is not None:
            people = self.get_ambulance_contents()
        else:
            people = getattr(self, 'ambulance_contents', [])
        # Add capacity info
        capacity_info = f'Ambulance Capacity = {len(self.__units)}'
        if not people:
            info = f'{capacity_info}\nThe ambulance is empty.'
        else:
            info = f'{capacity_info}\n' + '\n'.join(str(person) for person in people)
        tk.messagebox.showinfo('Ambulance Contents', info)

    def set_ambulance_contents(self, people):
        # Optional: allow setting ambulance contents directly
        self.ambulance_contents = people

    def render(self, max_cap, size):
        # Remove old units
        for unit in self.__units:
            self.canvas.delete(unit)
        self.__units = []
        tk.Label(self.canvas, text="Capacity", font=("Arial", 15)).place(x=100, y=0)

        # Always use 2 columns, stack extra dots below
        cols = 2
        rows = math.ceil(max_cap / 2)
        # --- Adjustable starting position for circles ---
        x_start = 131  # Change this to move the circles horizontally
        y_start = 192  # Change this to move the circles vertically
        # ----------------------------------------------
        x_gap = 60 - 27
        y_gap = size * 1.5
        idx = 0
        for row in range(rows):
            y = y_start + row * y_gap
            for col in range(cols):
                x = x_start + col * x_gap
                if idx < max_cap:
                    self.__units.append(self.create_circle(self.canvas, x, y, size))
                    idx += 1

    def update_fill(self, index):
        for i, unit in enumerate(self.__units):
            if i < index:
                self.canvas.itemconfig(unit, fill="midnightblue", outline="midnightblue")
            else:
                self.canvas.itemconfig(unit, fill="white", outline="gray")

    def create_circle(self, canvas, x, y, size):
        # Draw a circle (oval) with center (x, y)
        r = size // 2
        return canvas.create_oval(x, y, x + size, y + size, fill="white", outline="gray", width=2)
