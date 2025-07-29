import math
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk
import os
import tkinter.messagebox  # <-- Add this import
from ui_elements.theme import LABEL_FONT


class CapacityMeter(object):
    def __init__(self, root, w, h, max_cap, get_ambulance_contents=None):
        self.canvas = tk.Canvas(root, width=180, height=250, bg="#5B7B7A", highlightthickness=2, relief="groove", bd=2)
        # self.canvas = tk.Canvas(root, width)
        # Align with the clock (placed at x=math.floor(0.75 * w))
        # x_pos = math.floor(0.775 * w)
        # y_pos = math.floor(0.15 * h)  # move up by 20px
        x_pos = 1060
        y_pos = 190
        self.canvas.place(x=x_pos, y=y_pos)
        self.__units = []
        self.unit_size = 12  # Smaller dots for better proportion
        self.canvas.update()
        # Robust image path
        image_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'ambulance.png')
        self.bg_image = Image.open(image_path)
        canvas_width = int(self.canvas.winfo_width())
        canvas_height = int(self.canvas.winfo_height())
        # Scale image by 1.45x but keep aspect ratio
        max_width = int(canvas_width * 1.45)
        max_height = int(canvas_height * 1.45)
        img_w, img_h = self.bg_image.size
        scale = min(max_width / img_w, max_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        self.bg_image = self.bg_image.resize((new_w, new_h), Image.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)
        # Center the image in the canvas
        x_offset = (canvas_width - new_w) // 2
        y_offset = (canvas_height - new_h) // 2
        self.bg_image_id = self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.bg_photo)
        self.render(max_cap, self.unit_size)

        # Store the callback or data provider for ambulance contents
        self.get_ambulance_contents = get_ambulance_contents
        
        # Enable clicks after a short delay to prevent initialization dialogs
        self.canvas.after(1000, self._enable_clicks)
        
        # Bind click event to the canvas
        self.canvas.bind("<Button-1>", self.on_ambulance_click)

    def on_ambulance_click(self, event=None):
        # Only show dialog if this is a real user click (not during initialization)
        if event is None or not hasattr(self, '_click_enabled'):
            return
            
        # Get ambulance contents from callback or attribute
        if self.get_ambulance_contents is not None:
            people = self.get_ambulance_contents()
        else:
            people = getattr(self, 'ambulance_contents', [])
        # Count each type
        zombie_count = 0
        healthy_count = 0
        injured_count = 0
        for person in people:
            if person.get('class', '').lower() == 'zombie':
                zombie_count += 1
            elif person.get('class', '').lower() == 'default':
                if str(person.get('injured', '')).lower() == 'true':
                    injured_count += 1
                else:
                    healthy_count += 1
            elif person.get('original_status', '') == 'cured_zombie':
                # Treat cured humans as healthy humans
                healthy_count += 1
        capacity_info = f'Ambulance Capacity = {len(self.__units)}'
        info = f'{capacity_info}\n'
        info += f'Zombies: {zombie_count}\n'
        info += f'Healthy Humans: {healthy_count}\n'
        info += f'Injured Humans: {injured_count}\n'
        if (zombie_count + healthy_count + injured_count) == 0:
            info += 'The ambulance is empty.'
        tk.messagebox.showinfo('Ambulance Contents', info)

    def set_ambulance_contents(self, people):
        # Optional: allow setting ambulance contents directly
        self.ambulance_contents = people
        
    def _enable_clicks(self):
        # Enable click functionality after initialization delay
        self._click_enabled = True

    def render(self, max_cap, size):
        # Remove old units
        for unit in self.__units:
            self.canvas.delete(unit)
        self.__units = []
        # Draw the label directly on the canvas instead of using tk.Label
        capacity_font = font.Font(family="Fixedsys", size=14)
        canvas_width = int(self.canvas.winfo_width())
        self.canvas.create_text(canvas_width // 2, 15, text="Capacity", font= capacity_font, fill="white", anchor="center")

        # Always use 2 columns, stack extra dots below
        cols = 2
        rows = math.ceil(max_cap / 2)
        x_gap = 15  # Less space between columns
        y_gap = size * 1.2
        canvas_width = int(self.canvas.winfo_width())
        # Dynamically center the circles horizontally
        total_circles_width = (cols * size) + ((cols - 1) * x_gap)
        x_start = 74
        y_start = 120  # Keep dots below the label and ambulance
        idx = 0
        for row in range(rows):
            y = y_start + row * y_gap
            for col in range(cols):
                x = x_start + col * (size + x_gap)
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

    def reset_capacity(self, new_capacity):
        """
        Reset the capacity meter to show a new capacity value.
        This is used when upgrades are reset and the capacity returns to the base value.
        """
        # Re-render the capacity meter with the new capacity
        self.render(new_capacity, self.unit_size)
        # Reset the fill to show empty capacity
        self.update_fill(0)
        print(f"[DEBUG] Capacity meter reset to {new_capacity} slots")
