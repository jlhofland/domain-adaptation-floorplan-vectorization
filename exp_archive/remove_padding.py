import os
from PIL import Image

# Find all images in the 'data' folder that are called F1_unpadded.png. 
# When such an image exists remove the F1_original.png image and rename the F1_unpadded.png image to F1_original.png.
data_folder = 'data'
images = []

# Recursively walk through the data folder
for root, dirs, files in os.walk(data_folder):
    for file in files:
        if file == 'F1_unpadded.png':
            # Get the path of the F1_unpadded.png image
            unpadded_path = os.path.join(root, file)
            # Get the path of the F1_original.png image
            original_path = os.path.join(root, 'F1_original.png')
            # Remove the F1_original.png image
            os.remove(original_path)
            # Rename the F1_unpadded.png image to F1_original.png
            os.rename(unpadded_path, original_path)
            images.append(original_path)
            print(f"Renamed {original_path}")

print(f"Renamed {len(images)} images")


