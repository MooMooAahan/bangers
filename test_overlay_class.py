#!/usr/bin/env python3
"""
Test script to verify the AmbulanceOverlay class functionality.
"""

import tkinter as tk
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ui_elements.ambulance_overlay import AmbulanceOverlay
    print("✓ Successfully imported AmbulanceOverlay class")
except ImportError as e:
    print(f"✗ Failed to import AmbulanceOverlay: {e}")
    sys.exit(1)

def test_ambulance_overlay_class():
    """Test the AmbulanceOverlay class."""
    
    print("Creating test window...")
    
    # Create a test window
    root = tk.Tk()
    root.title("Ambulance Overlay Class Test")
    root.geometry("800x600")
    
    try:
        print("Creating AmbulanceOverlay instance...")
        overlay = AmbulanceOverlay(root, 800, 600)
        print("✓ AmbulanceOverlay instance created successfully")
        
        print("Placing overlay...")
        overlay.place(0, 0)
        print("✓ Overlay placed successfully")
        
        # Add a test label on top to verify layering
        test_label = tk.Label(root, text="Test Label - Should be on top of overlay", 
                             bg="red", fg="white", font=("Arial", 16))
        test_label.place(x=50, y=50)
        print("✓ Test label added on top")
        
        print("✓ All tests passed! The ambulance overlay should be visible with a red label on top.")
        print("Close the window to exit the test.")
        
        # Start the GUI
        root.mainloop()
        
    except Exception as e:
        print(f"✗ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ambulance_overlay_class() 