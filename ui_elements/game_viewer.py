import math
import tkinter as tk

from os.path import join
from PIL import ImageTk, Image


class GameViewer(object):
    def __init__(self, root, w, h, data_fp, humanoid):
        # Set image display size
        self.img_width = 256
        self.img_buffer = 30
        # Load the image to get its original height
        from PIL import Image
        img_path = join(data_fp, humanoid.fp)
        img = Image.open(img_path)
        orig_width, orig_height = img.size
        scale = min(self.img_width / orig_width, 1.0)  # Only downscale, never upscale
        self.img_display_width = int(orig_width * scale)
        self.img_display_height = int(orig_height * scale)
        canvas_height = self.img_display_height + self.img_buffer
        canvas_width = self.img_display_width
        self.canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
        self.canvas.update()
        self.photo = None
        self.create_photo(img_path)

    def delete_photo(self, event=None):
        self.canvas.delete('photo')

    def create_photo(self, fp):
        from PIL import Image
        img = Image.open(fp)
        orig_width, orig_height = img.size
        scale = min(self.img_width / orig_width, 1.0)
        display_width = int(orig_width * scale)
        display_height = int(orig_height * scale)
        img = img.resize((display_width, display_height), Image.LANCZOS)
        from PIL import ImageTk
        self.photo = ImageTk.PhotoImage(img)
        # Resize the canvas to fit the new image
        canvas_height = display_height + self.img_buffer
        canvas_width = display_width
        self.canvas.config(width=canvas_width, height=canvas_height)
        self.canvas.update()
        self.canvas.delete('photo')
        # Center the image in the canvas using calculated width
        x = (canvas_width - display_width) // 2
        y = self.img_buffer // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo, tags='photo')

    def display_score(self, score):
        self.canvas.delete('all')
        # Add a larger top margin so the text is not cut off
        top_margin = 50
        self.canvas.create_text(self.canvas.winfo_width()//2, top_margin, text="FINAL SCORE", font=("Arial", 30), anchor=tk.N)
        self.canvas.create_text(self.canvas.winfo_width()//2, top_margin+50, text="Killed {}".format(score["killed"]), font=("Arial", 15), anchor=tk.N)
        self.canvas.create_text(self.canvas.winfo_width()//2, top_margin+80, text="Saved {}".format(score["saved"]), font=("Arial", 15), anchor=tk.N)


def display_photo(img_path, w, h):
    img = Image.open(img_path)
    resized = img.resize((w, h), Image.LANCZOS)

    tk_img = ImageTk.PhotoImage(resized)
    return tk_img
