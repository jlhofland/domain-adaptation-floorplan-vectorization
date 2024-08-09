import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import tqdm

### INPUT ###

data_folder = 'data'
save_folder = 'paper/plots'
os.makedirs(save_folder, exist_ok=True)

fontsize    = 18

### STRUCUTRE ###

structure = {
    "Train": {
        "Source": "train_hqa_hq.txt",
        "Target": "train_c.txt"
    },
    "Validation": {
        "Source": "val_hqa_hq.txt",
        "Target": "val_c.txt"
    },
    "Test": {
        "Source": "test_hqa_hq.txt",
        "Target": "test_c.txt"
    }
}

class_structure = {
    "Rooms": {
        "ignore": ["Garage"],
        "labels": ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"],
        "colors": ["tab:blue", "tab:orange"]
    },
    "Icons": {
        "ignore": ["Fire Place", "Bathtub", "Chimney"],
        "labels": ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"],
        "colors": ["tab:blue", "tab:orange"]
    }
}

channel_dict = {"Red": [], "Green": [], "Blue": []}

### FUNCTIONS ###

def kl_divergence(P, Q):
    # Ensure P and Q are numpy arrays
    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)
    
    # Avoid division by zero and log of zero
    epsilon = 1e-10
    P += epsilon
    Q += epsilon
    
    # Calculate KL divergence
    return np.sum(P * np.log(P / Q))

def calculate_average_kl(vector_list):
    # List for KL divergences
    kl_list = []

    # Loop over list
    for i, v1 in enumerate(vector_list):
        for j, v2 in enumerate(vector_list):
            # Skip if same vector
            if i == j:
                continue

            # Calculate KL divergence
            kl = kl_divergence(v1, v2)

            # Append to list
            kl_list.append(kl)

    # Return the average of the KL divergences
    return np.mean(kl_list)

### PLOT ###
fig, ax = plt.subplots(figsize=(24, 4))

df_plot = []
for i, (split, split_domains) in enumerate(structure.items()):
    df = []
    for j, (class_type, class_data) in enumerate(class_structure.items()):
        print(f"Calculating KL divergence for {split} {class_type}...")

        # Create a dict for pixels
        pixels = {}

        # Loop over the domains
        for k, (domain, domain_data) in enumerate(split_domains.items()):
            # Create a dict for pixels with the structure of class_structure
            pixels[domain] = {class_label: {"Red": [], "Green": [], "Blue": []} for class_label in class_data["labels"]}

            count = 0

            # Loop through the data file
            with open(f'{data_folder}/{domain_data}', 'r') as f:
                # Loop using tqdm to show progress bar
                for line in tqdm.tqdm(f):
                    # Load data.pkl file
                    data = np.load(f'{data_folder}/{line.strip()}/data.pkl', allow_pickle=True)

                    # Extract
                    label = data['label']
                    image = data['image']

                    # Loop through the class structure and channels and append the pixels
                    for l, class_label in enumerate(class_data["labels"]):
                        for m, channel in enumerate(pixels[domain][class_label].keys()):
                            pixels[domain][class_label][channel].append(image[m, :, :][label[j] == l].flatten())

                    # if count == 10:
                    #     break
                    # count += 1

            # Calculate the KL divergence
            for class_label in class_data["labels"]:
                for channel in channel_dict.keys():
                    pixels[domain][class_label][channel] = np.histogram(np.concatenate(pixels[domain][class_label][channel]), bins=int(256/2), range=(0, 256), density=True)

        # Create a numpy array for the KL divergence
        kl_df = np.zeros((len(channel_dict), len(class_data["labels"])+1), dtype=float)
        for k, class_label in enumerate(class_data["labels"]):
            for l, channel in enumerate(channel_dict.keys()):
                kl_df[l, k] = kl_divergence(pixels["Source"][class_label][channel][0], pixels["Target"][class_label][channel][0])

        # Take channel mean
        kl_df = np.mean(kl_df, axis=0)

        # Replace inf with nan
        kl_df[kl_df == np.inf] = np.nan

        # Fill in class mean in the last column
        kl_df[-1] = np.nanmean(kl_df[:-1])

        # Append to the list
        df.append(kl_df)

    # Concat the items in the list to one numpy array (row)
    df = np.concatenate(df, axis=0)

    # Append to the list
    df_plot.append(df)

# Add another row for the mean of the classes
df_plot.append(np.mean(df_plot, axis=0))

np_array = np.array(df_plot)

# Plot the heatmap
cax = ax.imshow(np_array, cmap="viridis")

# Set the ticks
ax.set_xticks(np.arange(np_array.shape[1]))
ax.set_yticks(np.arange(np_array.shape[0]))

# Set the tick labels
ax.set_xticklabels(class_structure["Rooms"]["labels"] + ["Room Mean"] + class_structure["Icons"]["labels"] + ["Icon Mean"], fontsize=fontsize)
ax.set_yticklabels(list(structure.keys()) + ["Mean"], fontsize=fontsize)

# Set "Split Average" to bold
ax.get_yticklabels()[-1].set_fontweight("bold")

# Bold the "Room mean" 
ax.get_xticklabels()[len(class_structure["Rooms"]["labels"])].set_fontweight("bold")

# Bold the Icon mean
ax.get_xticklabels()[len(class_structure["Rooms"]["labels"] + ["Room Mean"] + class_structure["Icons"]["labels"])].set_fontweight("bold")

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add colorbar (close to the heatmap)
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

# Set the colorbar label
cbar.set_label("KL Divergence", fontsize=fontsize)

# Set the colorbar ticks
cbar.ax.tick_params(labelsize=fontsize)

# Loop over data dimensions and create text annotations
for i in range(np_array.shape[0]):
    for j in range(np_array.shape[1]):
        text = ax.text(j, i, f'{np_array[i, j]:.1f}', ha="center", va="center", color="w", fontsize=fontsize)

# Set the title
ax.set_title("Kullback-Leibner divergence from source to target domain", fontsize=fontsize)

# Save the plot
plt.savefig(f'{save_folder}/kl_divergence_heatmap.pdf', bbox_inches='tight')






                


