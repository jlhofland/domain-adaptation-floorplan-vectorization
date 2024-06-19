import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

# COLOR MAPS FOR ROOMS AND ICONS
cpool = ['#DCDCDC', '#b3de69', '#000000', '#8dd3c7', '#fdb462', # First was #DCDCDC
            '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969',
            '#577a4d', '#ffffb3']
cmap3 = colors.ListedColormap(cpool, 'rooms')
cm.register_cmap(cmap=cmap3)

cpool = ['#DCDCDC', '#8dd3c7', '#b15928', '#fdb462', '#ffff99',
            '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969',
            '#577a4d']
cmap3 = colors.ListedColormap(cpool, 'icons')
cm.register_cmap(cmap=cmap3)

# SETTINGS
data_folder = 'data'
files = ['train_c.txt', 'val_c.txt', 'test_c.txt']
# files = ['train_hqa_hq.txt', 'val_hqa_hq.txt', 'test_hqa_hq.txt']

room_classes = ["No Room", "Outdoor", "Wall", "Kitchen", "Living", "Bedroom", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
icon_classes = ["No Icon", "Window", "Door", "Closet", "Elec. Appl.", "Toilet", "Sink", "Sauna Bench", "Fireplace", "Bathtub", "Chimney"]

nr_rooms = 12
nr_icons = 11

# Create subplots of height 3 and width 2 with margin 0.1
fig, ax = plt.subplots(3, 2, figsize=(15, 9), gridspec_kw={'hspace': 0.25, 'wspace': 0.25})

# Add figure title
fig.suptitle('Distribution of rooms and icons for COLORFUL dataset (pixels)')

def get_unique(data_folder, file, dist_type='pixels'):
    # Create empty lists
    rooms_list = []
    icons_list = []

    # Open the file
    with open(f'{data_folder}/{file}', 'r') as f:
        # Loop using tqdm to show progress bar
        for line in tqdm.tqdm(f):
            # Load data.pkl file
            data = np.load(f'{data_folder}/{line.strip()}/data.pkl', allow_pickle=True)

            label = data['label']
            rooms = label[0]
            icons = label[1]

            if dist_type == 'pixels':
                # Filter 0 values
                rooms = rooms[rooms != 0]
                icons = icons[icons != 0]
            elif dist_type == 'classes':
                # Unique values
                rooms = np.unique(rooms)
                icons = np.unique(icons)
            else:
                raise ValueError(f'Unknown dist_type {dist_type}')

            # Add the rooms and icons to the list
            rooms_list.append(rooms)
            icons_list.append(icons)

    return np.concatenate(rooms_list), np.concatenate(icons_list)

for i, file in enumerate(files):
    # Get the unique rooms and icons
    rooms, icons = get_unique(data_folder, file)

    # Create histogram of rooms with the count of each room above the bars
    ax[i, 0].hist(rooms, bins=nr_rooms, range=(0, nr_rooms), align='left', rwidth=0.8, density=True)

    # Add the count of each room above the bars with a 45 degree rotation
    for j in range(nr_rooms):
        ax[i, 0].text(j, ax[i, 0].patches[j].get_height(), f'{ax[i, 0].patches[j].get_height() * 100:.1f}', ha='center', va='bottom')

    # Create histogram of icons
    ax[i, 1].hist(icons, bins=nr_icons, range=(0, nr_icons), align='left', rwidth=0.8, density=True)
    for j in range(nr_icons):
        ax[i, 1].text(j, ax[i, 1].patches[j].get_height(), f'{ax[i, 1].patches[j].get_height() * 100:.1f}', ha='center', va='bottom')

    # Set colors to the rooms
    room_colors = plt.cm.get_cmap('rooms')(np.linspace(0, 1, nr_rooms))
    for j, bar in enumerate(ax[i, 0].patches):
        bar.set_facecolor(room_colors[j])

    # Set colors to the icons
    icon_colors = plt.cm.get_cmap('icons')(np.linspace(0, 1, nr_icons))
    for j, bar in enumerate(ax[i, 1].patches):
        bar.set_facecolor(icon_colors[j])

    # Remove border around subplot
    ax[i, 0].spines['top'].set_visible(False)
    ax[i, 0].spines['right'].set_visible(False)
    ax[i, 1].spines['top'].set_visible(False)
    ax[i, 1].spines['right'].set_visible(False)

    # Remove x-axis labels
    ax[i, 0].set_xticks([])
    ax[i, 1].set_xticks([])

    # Set yaxis to percentage
    ax[i, 0].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax[i, 1].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

# Add labels at the top
ax[0, 0].set_title('Rooms')
ax[0, 1].set_title('Icons')

# Add row labels to the left
ax[0, 0].set_ylabel('Train')
ax[1, 0].set_ylabel('Validation')
ax[2, 0].set_ylabel('Test')

# Replace numbers of the x-axis with the room and icon classes
ax[2, 0].set_xticks(np.arange(nr_rooms))
ax[2, 0].set_xticklabels(room_classes, rotation=45, ha='right')
ax[2, 1].set_xticks(np.arange(nr_icons))
ax[2, 1].set_xticklabels(icon_classes, rotation=45, ha='right')

plt.show()
