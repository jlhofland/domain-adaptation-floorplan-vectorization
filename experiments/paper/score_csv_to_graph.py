import pandas as pd
import matplotlib.pyplot as plt
import os

# Type of classes to plot
class_type = "room"
experiment = "LC"
variation  = "Lambda (C)"

# Path to the CSV file
file_path = 'experiments\paper\csv\lambda\score_LC_room.csv'
save_path = 'experiments\paper\plots\lambda'

# Make sure the save path exists
os.makedirs(save_path, exist_ok=True)

# Load the CSV file
df = pd.read_csv(file_path)

# Extract relevant columns for plotting
lambda_values = df[variation].unique()
metrics = ['Class IoU', 'Class Acc']

# Create dictionaries to hold data for plotting
labels = ["Overall Acc", "Mean Acc", "FreqW Acc", "Mean IoU"]

# Create dictionaries to hold data for plotting
seg_data = {label: [] for label in labels}
vec_data = {label: [] for label in labels}

# Populate dictionaries with values
for label in labels:
    for lambda_val in lambda_values:
        row = df[(df[variation] == lambda_val)]
        seg_data[label].append(float(row[f'{label} (seg)'].values[0]))
        vec_data[label].append(float(row[f'{label} (vec)'].values[0]))

# Plotting Class IoU
plt.figure(figsize=(12, 6))
for label in labels:
    plt.plot(lambda_values, seg_data[label], marker='o', label=label)
plt.xlabel(variation)
plt.ylabel('Score (%)')
plt.title(f'{variation} vs Score: ' + class_type)
plt.legend()
plt.grid(True)
plt.savefig(f'{save_path}\{experiment}_{class_type}_score_seg.png')

# Plotting Class Acc
plt.figure(figsize=(12, 6))
for label in labels:
    plt.plot(lambda_values, vec_data[label], marker='o', label=label)
plt.xlabel(variation)
plt.ylabel('Score(%)')
plt.title(f'{variation} vs Score: ' + class_type)
plt.legend()
plt.grid(True)

# Save the plots
plt.savefig(f'{save_path}\{experiment}_{class_type}_score_vec.png')
