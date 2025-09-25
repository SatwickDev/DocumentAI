#!/usr/bin/env python3
"""
Create a simple test image for preprocessing
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image():
    """Create a simple test image with some text"""
    # Create a white image
    width, height = 800, 600
    image = Image.new('RGB', (width, height), 'white')
    
    # Get a drawing context
    draw = ImageDraw.Draw(image)
    
    # Add some text (simulating a document)
    try:
        # Try to use a default font, fallback to default if not available
        font = ImageFont.load_default()
    except:
        font = None
    
    # Add title
    draw.text((50, 50), "TEST DOCUMENT", fill='black', font=font)
    draw.text((50, 100), "This is a test document for preprocessing", fill='black', font=font)
    draw.text((50, 150), "It contains some text that should be", fill='black', font=font)
    draw.text((50, 200), "processed by the preprocessing service.", fill='black', font=font)
    
    # Add some lines and shapes to make it look more like a document
    draw.rectangle([50, 250, 750, 350], outline='black', width=2)
    draw.text((60, 270), "Section 1: Important Information", fill='black', font=font)
    draw.text((60, 300), "This section contains critical data", fill='black', font=font)
    draw.text((60, 320), "that needs to be processed accurately.", fill='black', font=font)
    
    # Add some noise to simulate a scanned document
    draw.rectangle([50, 400, 750, 500], outline='gray', width=1)
    draw.text((60, 420), "Section 2: Additional Notes", fill='gray', font=font)
    draw.text((60, 450), "Some text with varying quality", fill='darkgray', font=font)
    draw.text((60, 470), "to test preprocessing capabilities.", fill='gray', font=font)
    
    # Save the image
    filename = "test_document.png"
    image.save(filename)
    print(f"✅ Created test image: {filename}")
    return filename

if __name__ == "__main__":
    create_test_image()