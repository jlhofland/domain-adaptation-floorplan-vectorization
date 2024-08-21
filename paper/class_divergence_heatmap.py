import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import tqdm
import torch

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
    "Val": {
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
        "labels": ["No Icon", "Window", "Door", "Closet", "Elec. Appl." ,"Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"],
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
fig, ax = plt.subplots(2, 1,figsize=(24, 8), sharex='col', sharey='row', gridspec_kw={'hspace': 0.0, 'wspace': 0.0})

# Adjust hspace and wspace
plt.subplots_adjust(hspace=0, wspace=0)

df_plot = []
cl_plot = []
for i, (split, split_domains) in enumerate(structure.items()):
    df = []
    cl = []
    for j, (class_type, class_data) in enumerate(class_structure.items()):
        print(f"Calculating KL divergence for {split} {class_type}...")

        # Create a dict for pixels
        pixels = {}
        classes = {}

        # Loop over the domains
        for k, (domain, domain_data) in enumerate(split_domains.items()):
            # Create a dict for pixels with the structure of class_structure
            pixels[domain] = {class_label: {"Red": [], "Green": [], "Blue": []} for class_label in class_data["labels"]}
            classes[domain] = {class_label: 0 for class_label in class_data["labels"]}

            # Loop through the data file
            with open(f'{data_folder}/{domain_data}', 'r') as f:
                # Loop using tqdm to show progress bar
                for count, line in enumerate(tqdm.tqdm(f)):
                    # Load data.pkl file
                    data = np.load(f'{data_folder}/{line.strip()}/data.pkl', allow_pickle=True)

                    # Extract
                    label = data['label']
                    image = data['image']

                    # Loop through the class structure and channels and append the pixels
                    for l, class_label in enumerate(class_data["labels"]):
                        for m, channel in enumerate(pixels[domain][class_label].keys()):
                            pixels[domain][class_label][channel].append(image[m, :, :][label[j] == l].flatten())
                            classes[domain][class_label] += torch.sum(label[j] == l).item()

                    # if count == 100:
                    #     break

            # Calculate the KL divergence
            for class_label in class_data["labels"]:
                # Get density histograms
                for channel in channel_dict.keys():
                    pixels[domain][class_label][channel] = np.histogram(np.concatenate(pixels[domain][class_label][channel]), bins=int(256/2), range=(0, 256), density=True)

                # Normalize the data
                total = sum(classes[domain].values())
                for class_label in class_data["labels"]:
                    classes[domain][class_label] /= total

        # Create a numpy array for the KL divergence
        kl_df    = np.zeros((len(channel_dict), len(class_data["labels"])+1), dtype=float)
        diff_cls = np.zeros((len(class_data["labels"])+1), dtype=float)

        # Loop over the class labels
        for k, class_label in enumerate(class_data["labels"]):
            # Calculate the KL divergence
            for l, channel in enumerate(channel_dict.keys()):
                kl_df[l, k] = kl_divergence(pixels["Source"][class_label][channel][0], pixels["Target"][class_label][channel][0])

            # Calculate the relative percentage difference in classes
            diff_cls[k] = abs(classes["Source"][class_label] - classes["Target"][class_label])/(max(classes["Source"][class_label], classes["Target"][class_label]) + 1e-10)*100

        # Take channel mean
        kl_df = np.mean(kl_df, axis=0)

        # Replace inf with nan
        kl_df[kl_df == np.inf]       = np.nan
        diff_cls[diff_cls == np.inf] = np.nan

        # Fill in class mean in the last column
        kl_df[-1]    = np.nanmean(kl_df[:-1])
        diff_cls[-1] = np.nanmean(diff_cls[:-1])

        # Append to the list
        df.append(kl_df)
        cl.append(diff_cls)

    # Concat the items in the list to one numpy array (row)
    df = np.concatenate(df, axis=0)
    cl = np.concatenate(cl, axis=0)

    # Append to the list
    df_plot.append(df)
    cl_plot.append(cl)

# Add another row for the mean of the classes
df_plot.append(np.mean(df_plot, axis=0))
cl_plot.append(np.mean(cl_plot, axis=0))

# Assert they have the same shape
assert df_plot[0].shape == cl_plot[0].shape and df_plot[1].shape == cl_plot[1].shape

df_array = np.array(df_plot)
cl_array = np.array(cl_plot)

data = [df_array, cl_array]
cmap = ["viridis", "inferno"]
lbls = ["KL Divergence", "Relative Difference (%)"]
axlb = ["Input (pixels)", "Output (classes)"]

for a in range(2):
    # Plot the heatmap
    cax = ax[a].imshow(data[a], cmap=cmap[a])

    # Loop over data dimensions and create text annotations
    for i in range(data[a].shape[0]):
        for j in range(data[a].shape[1]):
            if (data[a][i, j] >= 0.9 and a == 0) or (data[a][i, j] >= 90 and a == 1):
                color = "black"
            else:
                color = "white"
            text = ax[a].text(j, i, f'{data[a][i, j]:.1f}' if a == 0 else f'{data[a][i, j]:.0f}', ha="center", va="center", color=color, fontsize=fontsize)

    # Set the ticks
    ax[a].set_xticks(np.arange(data[a].shape[1]))
    ax[a].set_yticks(np.arange(data[a].shape[0]))

    # Set the tick labels
    ax[a].set_xticklabels(class_structure["Rooms"]["labels"] + ["Room Mean"] + class_structure["Icons"]["labels"] + ["Icon Mean"], fontsize=fontsize)
    ax[a].set_yticklabels(list(structure.keys()) + ["Mean"], fontsize=fontsize)

    # Set yaxis labels
    ax[a].set_ylabel(axlb[a], fontsize=fontsize)

    # Set "Split Average" to bold
    ax[a].get_yticklabels()[-1].set_fontweight("bold")

    # Bold the "Room mean" 
    ax[-1].get_xticklabels()[len(class_structure["Rooms"]["labels"])].set_fontweight("bold")

    # Bold the Icon mean
    ax[-1].get_xticklabels()[len(class_structure["Rooms"]["labels"] + ["Room Mean"] + class_structure["Icons"]["labels"])].set_fontweight("bold")

    # Rotate the tick labels and set their alignment
    plt.setp(ax[a].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar (close to the heatmap)
    cbar = fig.colorbar(cax, ax=ax[a], fraction=0.015, pad=0.01)

    # Set the colorbar label
    cbar.set_label(lbls[a], fontsize=fontsize)

    # Set the colorbar ticks
    cbar.ax.tick_params(labelsize=fontsize)

# Set the title
ax[0].set_title("Adaptation feature gap from source to target domain", fontsize=fontsize, pad=20)

# Save the plot
plt.savefig(f'{save_folder}/kl_divergence_heatmap.pdf', bbox_inches='tight')






                


