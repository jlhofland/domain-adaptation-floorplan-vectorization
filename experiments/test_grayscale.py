import numpy as np
from PIL import Image

data_folder = 'data'
epsilons = [0.01, 0.1, 1]

for epsilon in epsilons:
    print(f'-- Checking for epsilon {epsilon}')
    # Reset grayscale count
    grayscale_count = 0
    count = 0
    percentages = []

    # Loop over lines in train_c.txt
    for line in open(f'{data_folder}/train_hqa_hq.txt').readlines():
        # Get the folder that is on the line
        folder = line.strip()

        # Open the image F1_original.png in the folder
        img = Image.open(f'{data_folder}/{folder}/F1_original.png')

        # if 3 channels, check channels to see if they are the same
        if img.mode == 'RGB' or img.mode == 'RGBA':
            # Add small epsilon to prevent rounding errors
            is_grayscale = np.all(np.abs(np.array(img)[:, :, 0] - np.array(img)[:, :, 1]) < epsilon) and np.all(np.abs(np.array(img)[:, :, 1] - np.array(img)[:, :, 2]) < epsilon)
        elif img.mode == 'L':
            is_grayscale = True
        else:
            raise ValueError(f'Image mode {img.mode} not supported')

        # Calculate percentage of grayscale images
        if is_grayscale:
            grayscale_count += 1
        else:
            # Create mask for indices where the difference is greater than epsilon
            mask1 = np.abs(np.array(img)[:, :, 0] - np.array(img)[:, :, 1]) > epsilon
            mask2 = np.abs(np.array(img)[:, :, 1] - np.array(img)[:, :, 2]) > epsilon

            # Combine masks
            mask = mask1 | mask2

            # Calulate the percentage of pixels that are not the same
            percentage = np.sum(mask) / (img.size[0] * img.size[1])

            # Append to list
            percentages.append(percentage)

        # Increment count
        count += 1
    
    # Print percentage of grayscale images (and average of pixels that are not the same)
    print(f'Grayscale images: {grayscale_count*100/count:.2f}%, Pixels: {np.mean(percentages)*100:.2f}%')
