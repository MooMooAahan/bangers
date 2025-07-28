#!/usr/bin/env python3
"""
Simple test to check if ambulance overlay image can be loaded.
"""

import os
from PIL import Image

def test_image_loading():
    """Test if the ambulance overlay image can be loaded."""
    
    print("Testing ambulance overlay image loading...")
    
    # Check if the image file exists
    image_path = os.path.join("images", "ambulance_perspective_overlay.png")
    print(f"Looking for image at: {image_path}")
    
    if os.path.exists(image_path):
        print(f"✓ Image file found!")
        
        try:
            # Try to load the image
            original_image = Image.open(image_path)
            print(f"✓ Image loaded successfully!")
            print(f"  - Size: {original_image.size}")
            print(f"  - Mode: {original_image.mode}")
            
            # Try to resize it
            resized_image = original_image.resize((800, 600), Image.Resampling.LANCZOS)
            print(f"✓ Image resized successfully to 800x600!")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading image: {e}")
            return False
    else:
        print(f"✗ Image file not found!")
        print("Available files in images/ directory:")
        try:
            for file in os.listdir("images"):
                print(f"  - {file}")
        except Exception as e:
            print(f"Error listing images directory: {e}")
        return False

if __name__ == "__main__":
    success = test_image_loading()
    if success:
        print("\n✓ All tests passed! The ambulance overlay should work.")
    else:
        print("\n✗ Tests failed. Check the errors above.") 