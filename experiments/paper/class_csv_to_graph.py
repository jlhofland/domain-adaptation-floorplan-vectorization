import pandas as pd
import matplotlib.pyplot as plt
import os

# Type of classes to plot
class_type = "room"
experiment = "LC"
variation  = "Lambda (C)"

# Path to the CSV file
file_path = 'experiments\paper\csv\lambda\class_LC_room.csv'
save_path = 'experiments\paper\plots'

# Make sure the save path exists
os.makedirs(save_path, exist_ok=True)

# Load the CSV file
df = pd.read_csv(file_path)

# Extract relevant columns for plotting
lambda_values = df[variation].unique()
metrics = ['Class IoU', 'Class Acc']

# Create dictionaries to hold data for plotting
icon_types = ['No Icon', 'Window', 'Door', 'Closet', 'Electrical Applience', 'Toilet', 'Sink', 'Sauna Bench']
room_types = ['Background', 'Outdoor', 'Wall', 'Kitchen', 'Living Room', 'Bed Room', 'Bath', 'Entry', 'Railing', 'Storage', 'Undefined']

labels = icon_types if class_type == 'icon' else room_types

# Create dictionaries to hold data for plotting
iou_data = {label: [] for label in labels}
acc_data = {label: [] for label in labels}

# Populate dictionaries with values
for icon in icon_types:
    for lambda_val in lambda_values:
        iou_row = df[(df[experiment] == lambda_val) & (df['metric'] == 'Class IoU')]
        acc_row = df[(df[experiment] == lambda_val) & (df['metric'] == 'Class Acc')]
        iou_data[icon].append(float(iou_row[f'{icon} (seg)'].values[0]))
        acc_data[icon].append(float(acc_row[f'{icon} (seg)'].values[0]))

# Plotting Class IoU
plt.figure(figsize=(12, 6))
for icon in icon_types:
    plt.plot(lambda_values, iou_data[icon], marker='o', label=icon)
plt.xlabel(experiment)
plt.ylabel('Class IoU (%)')
plt.title(f'{experiment} vs Class IoU: ' + class_type)
plt.legend()
plt.grid(True)
plt.savefig(f'{save_path}\{experiment}_{class_type}_class_iou.png')

# Plotting Class Acc
plt.figure(figsize=(12, 6))
for icon in icon_types:
    plt.plot(lambda_values, acc_data[icon], marker='o', label=icon)
plt.xlabel(experiment)
plt.ylabel('Class Acc (%)')
plt.title(f'{experiment} vs Class Acc: ' + class_type)
plt.legend()
plt.grid(True)

# Save the plots
plt.savefig(f'{save_path}\{experiment}_{class_type}_class_acc.png')
