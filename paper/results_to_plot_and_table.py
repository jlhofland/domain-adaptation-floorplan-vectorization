import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

### INPUT ###

experiment  = ""
data_folder = 'paper/results'
save_folder = 'paper/plots'
os.makedirs(save_folder, exist_ok=True)

experiments = {
    "LC": "Constant λ",
    "LF": "Finegrained λ",
    "LV": "Variable λ",
}
col_id_exps = 0 # First column is the experiment name and values are variations of the experiment
metric      = "Class IoU"
fontsize    = 18

### STRUCUTRE ###

class_structure = {
    "Rooms": {
        "data": {
            "Segmentation": "room_seg.csv",
            "Vectorization": "room_vec.csv",
            "Difference": ["room_seg.csv", "room_vec.csv"]
        },
        "ignore": ["Garage"],
        "labels": ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"],
        "colors": "tab:blue",
    },
    "Icons": {
        "data": {
            "Segmentation": "icon_seg.csv",
            "Vectorization": "icon_vec.csv",
            "Difference": ["icon_seg.csv", "icon_vec.csv"]
        },
        "ignore": ["Fire Place", "Bathtub", "Chimney"],
        "labels": ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"],
        "colors": "tab:orange",
    }
}

### PLOT ###
fig, ax = plt.subplots(1, len(experiments), figsize=(len(experiments)*8+4, 8))

for e, (exp, exp_name) in enumerate(experiments.items()):
    # Loop through the class structure
    for i, (class_type, class_data) in enumerate(class_structure.items()):
        for j, (data_type, data_file) in enumerate(class_data["data"].items()):
            if type(data_file) == list:
                # Deduct the second from the first (except first 2 columns)
                seg = pd.read_csv(f'{data_folder}/{exp}/{data_file[0]}')
                vec = pd.read_csv(f'{data_folder}/{exp}/{data_file[1]}')

                # Subtract the two dataframes and add back the first 2 columns
                df = vec.iloc[:, 2:]- seg.iloc[:, 2:]
                df.insert(0, vec.columns[0], vec.iloc[:, 0])
                df.insert(1, vec.columns[1], vec.iloc[:, 1])
            else:
                df = pd.read_csv(f'{data_folder}/{exp}/{data_file}')
            
            # Filter rows with the metric and remove the metric column
            df = df[df["metric"] == metric].drop(columns=["metric"])

            # Remove the colums in ignore
            df = df.drop(columns=class_data["ignore"])

            # Order by col_id_exps
            df = df.sort_values(by=df.columns[col_id_exps])

            # Add mean column (except first column)
            df["Mean"] = df.iloc[:, 1:].mean(axis=1)

            if data_type != "Difference":
                ax[e].plot(df.iloc[:, 0], df["Mean"], marker='o', label=f"{class_type} {data_type}", color=class_data["colors"], linestyle='--' if data_type == "Vectorization" else '-', markersize=10)

                # Plot the max value
                max_row = df[df["Mean"] == df["Mean"].max()]
                ax[e].scatter(max_row.iloc[:, 0], max_row["Mean"], color="tab:green", s=100, zorder=10, label="Max")

            # Add \textbf{} around the max value in a column (except first column) and put in 3 decimal places
            for column in df.columns[1:]:
                df[column] = df[column].apply(lambda x: f"\textbf{{{x:.3f}}}" if x == df[column].max() else f"{x:.3f}")

            # Make sure directory exists
            os.makedirs(f'{save_folder}/{exp}', exist_ok=True)

            # Save the dataframe to a latex table
            df.to_latex(f'{save_folder}/{exp}/{class_type}_{data_type}.tex', index=False, escape=False)

    # Set the labels
    ax[e].set_xlabel("Lambda (λ)", fontsize=fontsize)

    # Set the ticks to fontsize
    ax[e].tick_params(axis='both', which='major', labelsize=fontsize)

    # Set the title
    ax[e].set_title(exp_name, fontsize=fontsize)

ax[0].set_ylabel(metric, fontsize=fontsize)

# Get the labels and handles for the last
handles, labels = ax[-1].get_legend_handles_labels()
unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]

# Plot legend outside of the plot on the right
ax[-1].legend(*zip(*unique), fontsize=fontsize, loc='center left', bbox_to_anchor=(1, 0.5))

# Add room for the legends on the right side
plt.subplots_adjust(right=0.7)

# Tight layout
plt.tight_layout()

# Save the plot
plt.savefig(f'{save_folder}/{list(experiments.keys())}.pdf', bbox_inches='tight')

