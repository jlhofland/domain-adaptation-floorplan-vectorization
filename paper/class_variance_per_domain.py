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
    "High Quality Architectural": { 
        "Train": "train_hqa.txt",
        "Validation": "val_hqa.txt",
        "Test": "test_hqa.txt"
    },
    "High Quality": {
        "Train": "train_hq.txt",
        "Validation": "val_hq.txt",
        "Test": "test_hq.txt"
    },
    "Colored": {
        "Train": "train_c.txt",
        "Validation": "val_c.txt",
        "Test": "test_c.txt"
    }
}

class_structure = {
    "Rooms": {
        "ignore": ["Background"],
        "labels": ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"],
        "colors": ['#DCDCDC', '#b3de69', '#000000', '#8dd3c7', '#fdb462', '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969', '#577a4d', '#ffffb3'],
    },
    "Icons": {
        "ignore": ["No Icon"],
        "labels": ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"],
        "colors": ['#DCDCDC', '#8dd3c7', '#b15928', '#fdb462', '#ffff99', '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969', '#577a4d']
    }
}

# Remove the ignore labels
for class_type, class_data in class_structure.items():
    for ignore in class_data["ignore"]:
        class_data["labels"].remove(ignore)

### PLOT ###
fig, ax = plt.subplots(2, 4, figsize=(24, 8), sharey=True)

domain_dict = {domain: i for i, domain in enumerate(structure.keys())}
for i, (domain, domain_data) in enumerate(structure.items()):
    split_dict = {}
    for j, (split, split_data) in enumerate(domain_data.items()):
        # Initialize the split data and class data
        split_dict[split] = {}
        for class_type, class_data in class_structure.items():
            split_dict[split][class_type] = {class_name: 0 for class_name in class_data["labels"]}

        # Loop through the data file
        with open(f'{data_folder}/{split_data}', 'r') as f:
            # Loop using tqdm to show progress bar
            for line in tqdm.tqdm(f):
                # Load data.pkl file
                data = np.load(f'{data_folder}/{line.strip()}/data.pkl', allow_pickle=True)

                # Extract
                label = data['label']
                image = data['image']

                # Loop through the class structure
                for k, (class_type, class_data) in enumerate(class_structure.items()):
                    for l, class_name in enumerate(class_data["labels"]):
                        l += len(class_data["ignore"])
                        # Get the amount of pixels of class k
                        split_dict[split][class_type][class_name] += torch.sum(label[k] == l).item()

        # Normalize the data
        for class_type, class_data in class_structure.items():
            total = sum(split_dict[split][class_type].values())
            for class_name in class_data["labels"]:
                split_dict[split][class_type][class_name] /= total

    # Create a boxplot for each class type
    for j, (class_type, class_data) in enumerate(class_structure.items()):
        for k, class_name in enumerate(class_data["labels"]):
            # Get the data from each split
            data = [split_dict[split][class_type][class_name] for split in domain_data.keys()]

            # Plot the data
            ax[j, i].boxplot(data, positions=[k], widths=0.6, patch_artist=True, boxprops=dict(facecolor=class_data["colors"][k]), medianprops=dict(color='black'))

        # Set the labels
        if i == 0:
            ax[j, i].set_ylabel(class_type, fontsize=fontsize)

        if j == 0:
            ax[j, i].set_title(domain, fontsize=fontsize)

        # Set the ticks
        ax[j, i].set_xticks(range(len(class_data["labels"])))

        # Set the labels
        ax[j, i].set_xticklabels(class_data["labels"], rotation=45, fontsize=fontsize, ha='right')

    domain_dict[domain] = split_dict

# Also create boxplots for the total class variance (sum all domains)
for j, (class_type, class_data) in enumerate(class_structure.items()):
    for k, class_name in enumerate(class_data["labels"]):
        # Get the data from each split
        data = [[domain_dict[domain][split][class_type][class_name] for split in domain_dict[domain].keys()] for domain in domain_dict.keys()]

        # Flatten the data
        data = [item for sublist in data for item in sublist]

        # Plot the data
        ax[j, -1].boxplot(data, positions=[k], widths=0.6, patch_artist=True, boxprops=dict(facecolor=class_data["colors"][k]), medianprops=dict(color='black'))

    # Set the labels
    if j == 0:
        ax[j, -1].set_title("Total", fontsize=fontsize, fontweight='bold')

    # Set the ticks
    ax[j, -1].set_xticks(range(len(class_data["labels"])))

    # Set the labels
    ax[j, -1].set_xticklabels(class_data["labels"], rotation=45, fontsize=fontsize, ha='right')

plt.tight_layout()
plt.savefig(f'{save_folder}/class_variance_per_domain.pdf', bbox_inches='tight')
