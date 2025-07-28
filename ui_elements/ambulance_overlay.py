import tkinter as tk
from PIL import Image, ImageTk
import os

class AmbulanceOverlay:
    """
    A UI element that displays the ambulance view as an underlay
    for the main game interface. This can be modified and tweaked later.
    """
    
    def __init__(self, parent, width, height):
        """
        Initialize the ambulance overlay.
        
        Args:
            parent: The parent tkinter widget
            width: Width of the overlay
            height: Height of the overlay
        """
        self.parent = parent
        self.width = width
        self.height = height
        
        # Create a canvas for the overlay
        self.canvas = tk.Canvas(
            parent, 
            width=width, 
            height=height, 
            highlightthickness=0,
            bg="black"  # Fallback background
        )
        
        # Load and display the ambulance view
        self.load_overlay()
        
    def load_overlay(self):
        """Load and display the ambulance view image."""
        try:
            # Path to the ambulance view image
            image_path = os.path.join("images", "ambulance_perspective.png")
            
            # Load the original image
            original_image = Image.open(image_path)
            
            # Resize the image to fit the canvas dimensions
            resized_image = original_image.resize((self.width, self.height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage for tkinter
            self.overlay_image = ImageTk.PhotoImage(resized_image)
            
            # Display the image on the canvas
            self.canvas.create_image(0, 0, image=self.overlay_image, anchor="nw")
            
            print("Ambulance view loaded successfully")
            
        except Exception as e:
            print(f"Could not load ambulance view: {e}")
            # Fallback: create a simple colored background
            self.canvas.configure(bg="#1a1a2e")  # Dark blue-gray background
    
    def place(self, x, y):
        """Place the overlay at the specified coordinates."""
        self.canvas.place(x=x, y=y)
    
    def place_forget(self):
        """Remove the overlay from the display."""
        self.canvas.place_forget()
    
    def configure_opacity(self, opacity):
        """
        Configure the opacity of the overlay (0.0 to 1.0).
        Note: This is a placeholder for future opacity implementation.
        """
        # TODO: Implement opacity adjustment
        # This would require more complex image manipulation
        print(f"Opacity adjustment requested: {opacity}")
    
    def get_canvas(self):
        """Get the underlying canvas widget."""
        return self.canvas 