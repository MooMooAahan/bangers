import math
import tkinter as tk

from os.path import join
from PIL import ImageTk, Image


class GameViewer(object):
    def __init__(self, root, w, h, data_fp, image):
        ### Had to change some values here so when you retry it doesnt shrink the images 
        ### when we do the moral machine two image thingy we need to change this but thats for
        ### future somebody :P
        # Set image display size
        self.img_width = w  # Use the width parameter passed in (now 300)
        self.img_buffer = 30
        # Load the image to get its original height
        from PIL import Image
        img_path = join(data_fp, image.Filename) # adjusted to be image.Filename instead of humanoid.fp
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
        ##Output needed to be different winfo_width() doesnt give us same sized images
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


    def display_score(self, score,final_score=None, accuracy=None):
        tk.Label(self.canvas, text="FINAL SCORE", font=("Arial", 30)).pack(anchor=tk.NW)
        tk.Label(self.canvas, text="Killed {}".format(score["killed"]), font=("Arial", 15)).pack(anchor=tk.NW)
        tk.Label(self.canvas, text="Saved {}".format(score["saved"]), font=("Arial", 15)).pack(anchor=tk.NW)
        if final_score is not None:
            tk.Label(self.canvas, text="Final Score: {}".format(final_score), font=("Arial", 15)).pack(anchor=tk.NW)
        if accuracy is not None:
            tk.Label(self.canvas, text="Accuracy: {:.2f}%".format(accuracy), font=("Arial", 15)).pack(anchor=tk.NW)


def display_photo(img_path, w, h):
    img = Image.open(img_path)
    resized = img.resize((w, h), Image.LANCZOS)

    tk_img = ImageTk.PhotoImage(resized)
    return tk_img
