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
# files = ['train_c.txt', 'val_c.txt', 'test_c.txt']
files = [
    ['train_hqa_hq.txt', 'val_hqa_hq.txt', 'test_hqa_hq.txt'],
    ['train_c.txt', 'val_c.txt', 'test_c.txt'],
]

room_classes = ["No Room", "Outdoor", "Wall", "Kitchen", "Living", "Bedroom", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
icon_classes = ["No Icon", "Window", "Door", "Closet", "Elec. Appl.", "Toilet", "Sink", "Sauna Bench", "Fireplace", "Bathtub", "Chimney"]

nr_rooms = 12
nr_icons = 11

# Create subplots of height 3 and width 2 with margin 0.1
fig, ax = plt.subplots(4, 3, figsize=(24, 12), gridspec_kw={'hspace': 0.7, 'wspace': 0.2})

# Add figure title
fig.suptitle("Source vs target domain distribution (pixels)")

# Set the color map for the rooms
room_colors = plt.cm.get_cmap('rooms')(np.linspace(0, 1, nr_rooms))

# Set the color map for the icons
icon_colors = plt.cm.get_cmap('icons')(np.linspace(0, 1, nr_icons))

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

for i, type in enumerate(files):
    for j, file in enumerate(type):
        # Get unique rooms and icons
        rooms, icons = get_unique(data_folder, file)


        # Create histogram of rooms with the count of each room above the bars
        ax[i, j].hist(rooms, bins=nr_rooms, range=(0, nr_rooms), align='left', rwidth=0.8, density=True)
        for k in range(nr_rooms):
            ax[i, j].text(k, ax[i, j].patches[k].get_height(), f'{ax[i, j].patches[k].get_height() * 100:.1f}', ha='center', va='bottom')

        # Set the color map for the rooms
        for b, bar in enumerate(ax[i, j].patches):
            bar.set_facecolor(room_colors[b])

        # Create histogram of icons with the count of each icon above the bars
        ax[i+2, j].hist(icons, bins=nr_icons, range=(0, nr_icons), align='left', rwidth=0.8, density=True)
        for k in range(nr_icons):
            ax[i+2, j].text(k, ax[i+2, j].patches[k].get_height(), f'{ax[i+2, j].patches[k].get_height() * 100:.1f}', ha='center', va='bottom')

        # Set the color map for the icons
        for b, bar in enumerate(ax[i+2, j].patches):
            bar.set_facecolor(icon_colors[b])

        # Remove border around subplot
        ax[i, j].spines['top'].set_visible(False)
        ax[i+2, j].spines['top'].set_visible(False)
        ax[i, j].spines['right'].set_visible(False)
        ax[i+2, j].spines['right'].set_visible(False)

        # Remove x-axis labels
        ax[i, j].set_xticks([])
        ax[i+2, j].set_xticks([])

        # Set yaxis to percentage
        ax[i, j].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax[i+2, j].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

# Add labels at the top with margin between the title and the plots
ax[0, 0].set_title('Train')
ax[0, 1].set_title('Validation')
ax[0, 2].set_title('Test')

# Add row labels to the left
ax[0, 0].set_ylabel('Rooms (source)')
ax[1, 0].set_ylabel('Rooms (target)')
ax[2, 0].set_ylabel('Icons (source)')
ax[3, 0].set_ylabel('Icons (target)')

# Add margin top at ax[2, _]



# Replace numbers of the x-axis with the room and icon classes (only for the second row)
for i in range(3):
    ax[1, i].set_xticks(range(nr_rooms))
    ax[1, i].set_xticklabels(room_classes, rotation=45, ha='right')

    ax[3, i].set_xticks(range(nr_icons))
    ax[3, i].set_xticklabels(icon_classes, rotation=45, ha='right')

plt.show()
