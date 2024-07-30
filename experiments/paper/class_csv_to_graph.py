import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.cm as cm
from matplotlib import colors

# Type of classes to plot
class_type = "icon"
experiment = "LV"
variation  = "Lambda (V)"

# Path to the CSV file
file_path = f'experiments\paper\csv\lambda\{experiment}'
save_path = f'experiments\paper\plots\lambda\class\{experiment}'

# Update the font size
plt.rc('font', size=12)
plt.rc('axes', titlesize=12)

# Make sure the save path exists
os.makedirs(save_path, exist_ok=True)

# Load the CSV file
df = pd.read_csv(file_path + f'\class_{class_type}.csv')

# Extract relevant columns for plotting
lambda_values = df[variation].unique()
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

labels = icon_types if class_type == 'icon' else room_types
colorz = icon_colors if class_type == 'icon' else room_colors

# Create dictionaries to hold data for plotting
iou_data = {label: [] for label in labels}
acc_data = {label: [] for label in labels}

# Populate dictionaries with values
for icon in labels:
    for lambda_val in lambda_values:
        iou_row = df[(df[variation] == lambda_val) & (df['metric'] == 'Class IoU')]
        acc_row = df[(df[variation] == lambda_val) & (df['metric'] == 'Class Acc')]
        iou_data[icon].append(float(iou_row[f'{icon} (seg)'].values[0]))
        acc_data[icon].append(float(acc_row[f'{icon} (seg)'].values[0]))

# Plotting Class IoU
plt.figure(figsize=(24, 12))
for label in labels:
    # Plot IoU
    plt.plot(lambda_values, iou_data[label], marker='x', label=label, color=colorz[labels.index(label)])

    # Get max IoU
    max_iou = np.max(iou_data[label])

    # Get the lambda value for max IoU
    max_iou_lambda = lambda_values[np.argmax(iou_data[label])]
    
    # Plot a dot at the max IoU
    plt.scatter(max_iou_lambda, max_iou, color=colorz[labels.index(label)], s=100, marker='o')

# Plot mean IoU
mean_iou = np.mean([iou_data[label] for label in labels], axis=0)
plt.plot(lambda_values, mean_iou, marker='x', label='Mean', color='black', linestyle='dashed')

plt.xlabel(variation)
plt.ylabel('IoU')
plt.title(f'{class_type} IoU vs {variation}')
plt.legend()
plt.grid(True)
plt.savefig(f'{save_path}\iou_{class_type}.png')
