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
font_small  = 12

### STRUCUTRE ###

structure = {
    "A": { 
        "Train": "train_hqa.txt",
        "Validation": "val_hqa.txt",
        "Test": "test_hqa.txt"
    },
    "H": {
        "Train": "train_hq.txt",
        "Validation": "val_hq.txt",
        "Test": "test_hq.txt"
    },
    "C": {
        "Train": "train_c.txt",
        "Validation": "val_c.txt",
        "Test": "test_c.txt"
    }
}

class_structure = {
    "Rooms": {
        "ignore": ["Background"],
        "labels": ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"],
        "colors": ['#DCDCDC', '#b3de69', '#000000', '#8dd3c7', '#fdb462', '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969', '#577a4d', '#ffffb3', 'grey'] # Last color is for total
    },
    "Icons": {
        "ignore": ["No Icon"],
        "labels": ["No Icon", "Window", "Door", "Closet", "Elec. Appl." ,"Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"],
        "colors": ['#DCDCDC', '#8dd3c7', '#b15928', '#fdb462', '#ffff99', '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969', '#577a4d', 'grey'] # Last color is for total
    }
}
def format_float(value):
    if value < 10:
        return f"{value:.1f}"  # One decimal place for zero
    else:
        return f"{int(value)}"  # No decimal places for non-zero values
    
# Remove the ignore labels
for class_type, class_data in class_structure.items():
    for ignore in class_data["ignore"]:
        class_data["colors"].remove(class_data["colors"][class_data["labels"].index(ignore)])
        class_data["labels"].remove(ignore)

### PLOT ###
fig_rooms, ax_rooms = plt.subplots(1, len(class_structure["Rooms"]["labels"]), figsize=(24, 3), sharey=True)
fig_icons, ax_icons = plt.subplots(1, len(class_structure["Icons"]["labels"]), figsize=(24, 3), sharey=True)

subplots = {
    "Rooms": [fig_rooms, ax_rooms],
    "Icons": [fig_icons, ax_icons],
}

# Add horizontal lines at the y-ticks
for ax in ax_rooms:
    ax.yaxis.grid(True)
    # Set the y-ticks to 0-25 with steps of 5
    ax.set_yticks(np.arange(0, 27, 5))

for ax in ax_icons:
    ax.yaxis.grid(True)
    # Set the y-ticks to 0-30 with steps of 5
    ax.set_yticks(np.arange(0, 37, 5))

domain_dict = {domain: i for i, domain in enumerate(structure.keys())}
for i, (domain, domain_data) in enumerate(structure.items()):
    split_dict = {}
    for j, (split, split_data) in enumerate(domain_data.items()):
        # Initialize the split data and class data
        split_dict[split] = {}
        for class_type, class_data in class_structure.items():
            split_dict[split][class_type] = {class_name: [0, 0] for class_name in class_data["labels"]}

        # Loop through the data file
        with open(f'{data_folder}/{split_data}', 'r') as f:
            # Loop using tqdm to show progress bar
            for ln, line in enumerate(tqdm.tqdm(f)):
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
                        split_dict[split][class_type][class_name][0] += torch.sum(label[k] == l).item()
                        # Add one if the label contains at least one pixel
                        split_dict[split][class_type][class_name][1] += 1 if torch.sum(label[k] == l).item() > 0 else 0

                # if ln == 10:
                #     break

        # Normalize the data
        for class_type, class_data in class_structure.items():
            total = sum([split_dict[split][class_type][class_name][0] for class_name in class_data["labels"]])
            for k, class_name in enumerate(class_data["labels"]):
                split_dict[split][class_type][class_name][0] /= total / 100


    # Create a boxplot for each class type
    # for j, (class_type, class_data) in enumerate(class_structure.items()):
    #     # Plot the total variance in last subplot
    #     data = [split_dict[split][class_type][class_name][0] for split in domain_data.keys() for class_name in class_data["labels"]]

    #     # Plot the data
    #     subplots[class_type][1][-1].boxplot(data, positions=[i], widths=0.6, patch_artist=True, boxprops=dict(facecolor='grey'), medianprops=dict(color='black'))

    # Add the split data
    domain_dict[domain] = split_dict

# Create boxplot for each class (with the data of the the classes over all domains)
for j, (class_type, class_data) in enumerate(class_structure.items()):
    for k, class_name in enumerate(class_data["labels"]): # + [class_type]
        if class_name == class_type:
            data = [domain_dict[domain][split][class_type][class_name][0] for domain in structure.keys() for split in structure[domain].keys() for class_name in class_data["labels"]]
            samp = sum([domain_dict[domain][split][class_type][class_name][1] for domain in structure.keys() for split in structure[domain].keys() for class_name in class_data["labels"]])
            mean = np.mean(data)
        else:
            for l, (domain, split_dict) in enumerate(domain_dict.items()):
                # Get the data from each split
                data = [split_dict[split][class_type][class_name][0] for split in split_dict.keys()]
                samp = sum([split_dict[split][class_type][class_name][1] for split in split_dict.keys()])

                # Plot the data
                subplots[class_type][1][k].boxplot(data, positions=[l], widths=0.6, patch_artist=True, boxprops=dict(facecolor=class_data["colors"][k]), medianprops=dict(color='black'))

            data = [domain_dict[domain][split][class_type][class_name][0] for domain in structure.keys() for split in structure[domain].keys()]
            samp = sum([domain_dict[domain][split][class_type][class_name][1] for domain in structure.keys() for split in structure[domain].keys()])

        # Plot the data
        subplots[class_type][1][k].boxplot(data, positions=[len(structure)], widths=0.6, patch_artist=True, boxprops=dict(facecolor=class_data["colors"][k]), medianprops=dict(color='black'))

        # Set the title to the class name at the top of the box, but add some space between the top of the box and the title
        subplots[class_type][1][k].set_title(class_name, fontsize=fontsize, pad=20)

        for l, (domain, split_dict) in enumerate(domain_dict.items()):
            # Set the ticks
            subplots[class_type][1][k].set_xticks(range(len(structure)+1))

            label = [f'{domain}' for domain in structure.keys()] + [f'T']

            # Set the labels
            subplots[class_type][1][k].set_xticklabels(label, fontsize=fontsize)

            # Set last tick to bold
            subplots[class_type][1][k].get_xticklabels()[-1].set_fontweight('bold')

            # Set y to fontsize
            subplots[class_type][1][k].tick_params(axis='y', labelsize=fontsize)

# for i, (domain, domain_data) in enumerate(structure.items()):
#     for j, (class_type, class_data) in enumerate(class_structure.items()):
#         # Plot the total variance in last subplot
#         data = [domain_dict[domain][split][class_type][class_name][0] for split in domain_data.keys() for class_name in class_data["labels"]]
#         samp = sum([domain_dict[domain][split][class_type][class_name][1] for split in domain_data.keys() for class_name in class_data["labels"]])

#         # Add the samp above the boxplot
#         subplots[class_type][1][-1].text(i, subplots[class_type][1][-1].get_ybound()[1] + 0.01, f'{np.mean(data):.1f}', ha='center', va='bottom', fontsize=fontsize/2)

# Handle texts
for j, (class_type, class_data) in enumerate(class_structure.items()):
    for k, class_name in enumerate(class_data["labels"]):
        for l, (domain, split_dict) in enumerate(domain_dict.items()):
            # Get the data from each split
            data = [split_dict[split][class_type][class_name][0] for split in split_dict.keys()]

            # Set text
            subplots[class_type][1][k].text(l, subplots[class_type][1][k].get_ybound()[1] + 0.01, f'{np.mean(data):.1f}', ha='center', va='bottom', fontsize=font_small)

        # Get the total data
        data = [domain_dict[domain][split][class_type][class_name][0] for domain in structure.keys() for split in structure[domain].keys()]

        # Set text
        subplots[class_type][1][k].text(len(structure), subplots[class_type][1][k].get_ybound()[1] + 0.01, f'{np.mean(data):.1f}', ha='center', va='bottom', fontsize=font_small, fontweight='bold')

# Set y-labels
ax_rooms[0].set_ylabel("Density (%)", fontsize=fontsize)
ax_icons[0].set_ylabel("Density (%)", fontsize=fontsize)

# Tight layout
fig_rooms.tight_layout()
fig_icons.tight_layout()

# Save the plots
fig_rooms.savefig(f'{save_folder}/domain_variance_per_class_rooms.pdf', bbox_inches='tight')
fig_icons.savefig(f'{save_folder}/domain_variance_per_class_icons.pdf', bbox_inches='tight')
