import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.widgets import Slider
import os

# ignore MatplotlibDeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def discrete_cmap():
    """create a colormap with N (N<15) discrete colors and register it"""
    # define individual colors as hex values
    cpool = ['#ffffff', '#b3de69', '#000000', '#8dd3c7', '#fdb462', # First was #DCDCDC
             '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969',
             '#577a4d', '#ffffb3']
    cmap3 = colors.ListedColormap(cpool, 'rooms')
    cm.register_cmap(cmap=cmap3)

    cpool = ['#ffffff', '#8dd3c7', '#b15928', '#fdb462', '#ffff99',
             '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969',
             '#577a4d']
    cmap3 = colors.ListedColormap(cpool, 'icons')
    cm.register_cmap(cmap=cmap3)

    """create a colormap with N (N<15) discrete colors and register it"""
    # define individual colors as hex values
    cpool = ['#DCDCDC', '#b3de69', '#000000', '#8dd3c7', '#fdb462',
             '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969',
             '#577a4d', '#ffffb3', 'd3d5d7']
    cmap3 = colors.ListedColormap(cpool, 'rooms_furu')
    cm.register_cmap(cmap=cmap3)

discrete_cmap()

room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]

data_folder = 'data'
plot_folder = 'experiments/plots'
file_list   = ['train_hqa_hq.txt', 'val_hqa_hq.txt']

n_rooms = 12
n_icons = 11

count = 0

for file in file_list:
    # Loop over lines in the file
    with open(f'{data_folder}/{file}', 'r') as f:
        for line in f:
            # Load data.pkl file
            data = np.load(f'{data_folder}/{line.strip()}/data.pkl', allow_pickle=True)

            label    = data['label']
            image    = data['image']

            rooms    = label[0]
            icons    = label[1]

            # Normalize the image
            image = 2 * (image / 255.0) - 1

            # Move axis of the image and scale pixel values
            np_img = np.moveaxis(image.numpy(), 0, -1) / 2 + 0.5

            # Create a 2 x 1 grid of plot to show the rooms and icons next to eachother
            fig, ax = plt.subplots(1, 2, figsize=(40, 20))

            # Set background color to white
            fig.patch.set_facecolor('white')

            # Plot the rooms
            ax[0].imshow(np_img)
            rp = ax[0].imshow(rooms, alpha=0.5, cmap='rooms', vmin=0, vmax=n_rooms-0.1)
            ax[0].set_title('Rooms')
            ax[0].axis('off')

            # Add legenda for the rooms
            cbar = plt.colorbar(rp, ticks=np.arange(n_rooms) + 0.5, fraction=0.046, pad=0.01)
            cbar.ax.set_yticklabels(room_classes, fontsize=20)

            # Plot the icons
            ax[1].imshow(np_img)
            ip = ax[1].imshow(icons, alpha=0.5, cmap='icons', vmin=0, vmax=n_icons-0.1)
            ax[1].set_title('Icons')
            ax[1].axis('off')

            # Add legenda for the icons
            cbar = plt.colorbar(ip, ticks=np.arange(n_icons) + 0.5, fraction=0.046, pad=0.01)
            cbar.ax.set_yticklabels(icon_classes, fontsize=20)

            # Add text that I can copy as title
            plt.suptitle(line.strip(), fontsize=20)

            # Format line.strip() to filename
            line = line.strip().replace('/', '')

            # Format file as foldername (by removing.txt)
            folder = file.replace('.txt', '')

            # Make sure folder exists
            if not os.path.exists(f'{plot_folder}/{folder}'):
                os.makedirs(f'{plot_folder}/{folder}')

            # Save the figure to plots folder
            plt.savefig(f'{plot_folder}/{folder}/{line}.png', bbox_inches='tight')
            plt.close()
            
