import numpy as np
from PIL import Image
from PIL import ImageOps
import os

data_folder = 'data'
count = 0

# Function that pads the image with a border 1/2 of the required padding on each side
def pad_image(img):
    # Get the resolution of the image
    resolution = img.size

    # Calculate the padding
    padding = (max(0, 256 - resolution[0]) / 2, max(0, 256 - resolution[1]) / 2)

    # make sure padding is an integer (round up if necessary)
    padding = tuple([int(np.ceil(p)) for p in padding])

    # Pad the image (1/2 padding on each side)
    img = ImageOps.expand(img, border=padding, fill='white')

    return img


# Loop over the files
for file in ['val_hqa_hq.txt', 'test_hqa.txt', 'test_hq.txt', 'test_c.txt']:
    # Print the file we are checking
    print(f"-- Checking for < 256x256 images in {file}")

    # Loop over lines in train_hqa_hq.txt
    for line in open(f'{data_folder}/{file}').readlines():
        # Open the image using PIL
        img = Image.open(f'{data_folder}/{line.strip()}/F1_original.png')

        # Get resolution
        resolution = img.size

        # Check if resolution is smaller than 256x256
        if resolution[0] < 256 or resolution[1] < 256:
            # Pad the image
            pad = pad_image(img)

            # Save the image and the padded image
            img.save(f'{data_folder}/{line.strip()}/F1_unpadded.png')
            pad.save(f'{data_folder}/{line.strip()}/F1_original.png')

            # Print resizing
            print(f'{line.strip()} had resolution {resolution} and was resized to {pad.size}')
            count += 1

print(f"Resized {count} images")