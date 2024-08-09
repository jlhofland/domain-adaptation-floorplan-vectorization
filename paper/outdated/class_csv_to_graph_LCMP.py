import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.cm as cm
from matplotlib import colors

# Type of classes to plot
experiment = "LC_MP"
variation  = "Latent Size"

# Path to the CSV file
file_path = f'experiments\paper\csv\lambda\{experiment}'
save_path = f'experiments\paper\plots\lambda\class\{experiment}'

# Make sure the save path exists
os.makedirs(save_path, exist_ok=True)

# Load the CSV file
rooms = pd.read_csv(file_path + '\class_room.csv')
icons = pd.read_csv(file_path + '\class_icon.csv')

# Rename columns ['runconfig[mmd\.latent_channels', 'runconfig[mmd\.latent_transformation]'] -> ['Channels', 'MP']
rooms.rename(columns={'runconfig[mmd\.latent_channels]': 'Channels', 'runconfig[mmd\.latent_transformation]': 'MP'}, inplace=True)
icons.rename(columns={'runconfig[mmd\.latent_channels]': 'Channels', 'runconfig[mmd\.latent_transformation]': 'MP'}, inplace=True)

# Change MP to int (nn.AdaptiveMaxPool2d((x,x)) -> x)
rooms['MP'] = rooms['MP'].apply(lambda x: int(x.split(',')[1].split(')')[0]))
icons['MP'] = icons['MP'].apply(lambda x: int(x.split(',')[1].split(')')[0]))

# Values that the col 'metric' can take
metrics = ['Class IoU', 'Class Acc']

# Create dictionaries to hold data for plotting
icon_types = ['No Icon', 'Window', 'Door', 'Closet', 'Electrical Applience', 'Toilet', 'Sink', 'Sauna Bench']
room_types = ['Background', 'Outdoor', 'Wall', 'Kitchen', 'Living Room', 'Bed Room', 'Bath', 'Entry', 'Railing', 'Storage', 'Undefined']

# ALL ROOM COLORS
cpool = ['#DCDCDC', '#b3de69', '#000000', '#8dd3c7', '#fdb462', '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969', '#577a4d', '#ffffb3']

# Remove index 10
cpool.pop(10) # Garage

# Create a new color map
cmap3 = colors.ListedColormap(cpool, 'rooms')
cm.register_cmap(cmap=cmap3)

# ALL ICON COLORS
cpool = ['#DCDCDC', '#8dd3c7', '#b15928', '#fdb462', '#ffff99', '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969','#577a4d']

# Remove indexes [8, 9, 10]
cpool.pop(10) # Garage
cpool.pop(9)  # Bathtub
cpool.pop(8)  # Fireplace

# Create a new color map
cmap3 = colors.ListedColormap(cpool, 'icons')
cm.register_cmap(cmap=cmap3)

# Get the colors for the rooms and icons
room_colors = plt.cm.get_cmap('rooms')(np.linspace(0, 1, len(room_types)))
icon_colors = plt.cm.get_cmap('icons')(np.linspace(0, 1, len(icon_types)))

# Create new column by multiplying the MP^2 by the Channels
rooms['LS'] = rooms['MP'] * rooms['MP'] * rooms['Channels']
icons['LS'] = icons['MP'] * icons['MP'] * icons['Channels']

# Get unique values for Lambda (C) 
lambdas = rooms['Lambda (C)'].unique()

# Create figure for rooms and icons of width lambdas
fig_room, ax_room = plt.subplots(1, len(lambdas), figsize=(24, 12))
fig_icon, ax_icon = plt.subplots(1, len(lambdas), figsize=(24, 12))

# Update the font size
plt.rc('font', size=12)
plt.rc('axes', titlesize=12)

for i, lm in enumerate(lambdas):
    icon_data = icons[icons['Lambda (C)'] == lm]
    room_data = rooms[rooms['Lambda (C)'] == lm]

    # Get rows where the metric is 'Class IoU'
    icon_data_iou = icon_data[icon_data['metric'] == 'Class IoU'].sort_values(by='LS')
    room_data_iou = room_data[room_data['metric'] == 'Class IoU'].sort_values(by='LS')

    iou_list = pd.DataFrame()

    # Plot the data, the x axis should be the LS, the y axis should be percentage, and the each line should be a different class
    for j, icon in enumerate(icon_types):
        ax_icon[i].plot(icon_data_iou['LS'], icon_data_iou[f'{icon} (seg)'], label=icon, color=icon_colors[j], marker='x')

        # Get the values of iou of this clas
        iou_list[icon] = icon_data_iou[f'{icon} (seg)']

        # Highlight the best value
        best = icon_data_iou[f'{icon} (seg)'].max()

        # Get the LS of the best value
        best_ls = icon_data_iou[icon_data_iou[f'{icon} (seg)'] == best]['LS'].values[0]

        # Show a dot in the plot at the best LS
        ax_icon[i].scatter(best_ls, best, color=icon_colors[j], s=100, marker='o')

    # Calculate the mean of the iou for this lambda
    iou_list['Mean'] = iou_list.mean(axis=1)

    # Plot the mean of the iou
    ax_icon[i].plot(icon_data_iou['LS'], iou_list['Mean'], label='Mean', linestyle='--', color='black', marker='x')
    
    # Set the title and legend
    ax_icon[i].set_title(f'Lambda {lm}')

    # Set x and y labels
    ax_icon[i].set_xticks(icon_data_iou['LS'])
    ax_icon[i].set_xlabel('Latent Size')
    ax_icon[i].set_ylabel('IoU')

    iou_list = pd.DataFrame()

    for j, room in enumerate(room_types):
        ax_room[i].plot(room_data_iou['LS'], room_data_iou[f'{room} (seg)'], label=room, color=room_colors[j], marker='x')

        # Get the values of iou of this clas
        iou_list[room] = room_data_iou[f'{room} (seg)']

        # Highlight the best value
        best = room_data_iou[f'{room} (seg)'].max()

        # Get the LS of the best value
        best_ls = room_data_iou[room_data_iou[f'{room} (seg)'] == best]['LS'].values[0]

        # Show a dot in the plot at the best LS
        ax_room[i].scatter(best_ls, best, color=room_colors[j], s=100, marker='o')

    # Calculate the mean of the iou for this lambda
    iou_list['Mean'] = iou_list.mean(axis=1)

    # Plot the mean of the iou
    ax_room[i].plot(room_data_iou['LS'], iou_list['Mean'], label='Mean', linestyle='--', color='black', marker='x')

    # Set the title and legend
    ax_room[i].set_title(f'Lambda {lm}')

    # Set x and y labels
    ax_room[i].set_xticks(room_data_iou['LS'])
    ax_room[i].set_xlabel('Latent Size')
    ax_room[i].set_ylabel('IoU (%)')

    if (i == 0):
        fig_room.legend(loc='center right')
        fig_icon.legend(loc='center right')

# Set title
fig_room.suptitle(f'Room IoU vs {variation}')
fig_icon.suptitle(f'Icon IoU vs {variation}')


# Save the figure
fig_icon.savefig(f'{save_path}\iou_icon.png')
fig_room.savefig(f'{save_path}\iou_room.png')

                        

