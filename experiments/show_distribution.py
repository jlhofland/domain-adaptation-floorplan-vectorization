import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm

data_folder = 'data'
file = 'train_hqa_hq.txt'

room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience", "Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]

nr_rooms = 12
nr_icons = 11

rooms_list = []
icons_list = []

with open(f'{data_folder}/{file}', 'r') as f:
    # Loop using tqdm to show progress bar
    for line in tqdm.tqdm(f):
        # Load data.pkl file
        data = np.load(f'{data_folder}/{line.strip()}/data.pkl', allow_pickle=True)

        label = data['label']
        rooms = label[0]
        icons = label[1]

        # Filter 0 values
        rooms = rooms[rooms != 0]
        icons = icons[icons != 0]

        # Add the rooms and icons to the list
        rooms_list.append(rooms)
        icons_list.append(icons)

# Concatenate lists to flatten them
rooms_flat = np.concatenate(rooms_list)
icons_flat = np.concatenate(icons_list)

# Plot the label distribution of rooms and icons using histograms
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

# Plot the rooms with the labels (rooms_classes) as x-axis
ax[0].hist(rooms_flat, bins=nr_rooms, range=(0, nr_rooms-0.1), align='left', rwidth=0.8)
ax[0].set_title('Rooms')
ax[0].set_xlabel('Room class')

# Replace numerical values with room_classes
ax[0].set_xticks(np.arange(nr_rooms))
ax[0].set_xticklabels(room_classes, rotation=45)
ax[0].set_xlim(0, nr_rooms)

# Plot the icons with the labels (icon_classes) as x-axis
ax[1].hist(icons_flat, bins=nr_icons, range=(0, nr_icons-0.1), align='left', rwidth=0.8)
ax[1].set_title('Icons')
ax[1].set_xlabel('Icon class')

# Replace numerical values with icon_classes
ax[1].set_xticks(np.arange(nr_icons))
ax[1].set_xticklabels(icon_classes, rotation=45)
ax[1].set_xlim(0, nr_icons)

plt.show()
