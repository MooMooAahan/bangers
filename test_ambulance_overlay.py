#!/usr/bin/env python3
"""
Simple test script to verify the ambulance overlay functionality.
"""

import tkinter as tk
from ui_elements.ambulance_overlay import AmbulanceOverlay
import os

def test_ambulance_overlay():
    """Test the ambulance overlay in a simple window."""
    
    # Create a test window
    root = tk.Tk()
    root.title("Ambulance Overlay Test")
    root.geometry("800x600")
    
    # Check if the image file exists
    image_path = os.path.join("images", "ambulance_perspective_overlay.png")
    if os.path.exists(image_path):
        print(f"✓ Ambulance overlay image found at: {image_path}")
    else:
        print(f"✗ Ambulance overlay image not found at: {image_path}")
        print("Available files in images/ directory:")
        try:
            for file in os.listdir("images"):
                print(f"  - {file}")
        except Exception as e:
            print(f"Error listing images directory: {e}")
        return
    
    # Create the ambulance overlay
    try:
        print("Creating ambulance overlay...")
        overlay = AmbulanceOverlay(root, 800, 600)
        overlay.place(0, 0)
        print("✓ Ambulance overlay created successfully")
        
        # Add a test label on top to verify layering
        test_label = tk.Label(root, text="Test Label - Should be on top of overlay", 
                             bg="red", fg="white", font=("Arial", 16))
        test_label.place(x=50, y=50)
        
        print("✓ Test label added on top")
        print("If you can see the ambulance overlay image with a red label on top, the test is successful!")
        
    except Exception as e:
        print(f"✗ Error creating ambulance overlay: {e}")
        import traceback
        traceback.print_exc()
    
    # Start the GUI
    print("Starting test window...")
    root.mainloop()

if __name__ == "__main__":
    test_ambulance_overlay() 