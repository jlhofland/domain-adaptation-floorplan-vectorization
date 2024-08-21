import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

### INPUT ###

experiment  = ""
data_folder = 'paper/results'
save_folder = 'paper/plots'
os.makedirs(save_folder, exist_ok=True)

col_id_exps = 0 # First column is the experiment name and values are variations of the experiment
metric      = "Class IoU"
fontsize    = 18
ignore_cols = ["Lambda (V)"]

### FUNCTIONS ###

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
        "colors": ["tab:cyan", "tab:red"]
    },
    "Icons": {
        "data": {
            "Segmentation": "icon_seg.csv",
            "Vectorization": "icon_vec.csv",
            "Difference": ["icon_seg.csv", "icon_vec.csv"]
        },
        "ignore": ["Fire Place", "Bathtub", "Chimney"],
        "labels": ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"],
        "colors": ["tab:cyan", "tab:red"]
    }
}

experiments = {
    "MP001": {
        "Channels": ["256", "512"],
        "Name": "λ = 0.01"
    },
    "MP01": {
        "Channels": ["256", "512"],
        "Name": "λ = 0.1"
    },
}

### PLOT ###

fig, ax = plt.subplots(1*len(experiments), len(class_structure), figsize=(len(class_structure)*4+4, 12), sharey='row', sharex=True, gridspec_kw={'hspace': 0.0, 'wspace': 0.0})

for i, (exp, exp_data) in enumerate(experiments.items()):
    channels = exp_data["Channels"]
    plot_shift = i*len(class_structure)
    # Iterate over the class structure
    for j, (class_name, class_data) in enumerate(class_structure.items()):
        for l, (data_type, data_file) in enumerate(class_data["data"].items()):
            class_frames = []
            for k, channel in enumerate(channels):  
                if type(data_file) == list:
                    # Read both data files
                    seg = pd.read_csv(f'{data_folder}/{exp}/{channel}/{data_file[0]}')
                    vec = pd.read_csv(f'{data_folder}/{exp}/{channel}/{data_file[1]}')

                    # Drop the columns in ignore
                    if set(ignore_cols).issubset(set(seg.columns)):
                        seg = seg.drop(columns=ignore_cols)
                    if set(ignore_cols).issubset(set(vec.columns)):
                        vec = vec.drop(columns=ignore_cols)

                    # Subtract the two dataframes and add back the first 2 columns
                    df = vec.iloc[:, 2:]- seg.iloc[:, 2:]
                    df.insert(0, vec.columns[0], vec.iloc[:, 0])
                    df.insert(1, vec.columns[1], vec.iloc[:, 1])
                else:
                    # Read the data
                    df = pd.read_csv(f'{data_folder}/{exp}/{channel}/{data_file}')

                    # Drop the columns in ignore
                    if set(ignore_cols).issubset(set(df.columns)):
                        df = df.drop(columns=ignore_cols)

                # Filter rows with the metric and remove the metric column
                df = df[df["metric"] == metric].drop(columns=["metric"])

                # Remove the colums of classes we want to ignore
                df = df.drop(columns=class_data["ignore"])

                # Order by col_id_exps
                df = df.sort_values(by=df.columns[col_id_exps])

                # Add mean column (except first column)
                df["Mean"] = df.iloc[:, 1:].mean(axis=1)

                # Get row where first col is NaN
                nan_row = df[df.iloc[:, 0].isna()]

                # Remove the row with NaN
                df = df.drop(nan_row.index)

                if data_type != "Difference":
                    # Plot a horizontal line at the Mean of the NaN row with marker x and color gray
                    ax[j][i].axhline(y=nan_row["Mean"].values[0]*100, color='gray', linestyle='dotted', zorder=9, label='No adaptation') 
                    
                    # Plot the data
                    ax[j][i].plot(df.iloc[:, 0], df["Mean"]*100, label=f'{channel} {data_type}', color=class_data["colors"][k], linestyle='-' if data_type == "Segmentation" else '--', marker='o', markersize=10)

                    # Plot the max value
                    max_row = df[df["Mean"] == df["Mean"].max()]
                    ax[j][i].scatter(max_row.iloc[:, 0], max_row["Mean"]*100, color='tab:green', zorder=10, label='Max', s=100)

                # Add a column after first column with the channel
                df.insert(0, "Channels", channel)

                # Add
                class_frames.append(df)

            # Merge the dataframes
            df_merged = pd.concat(class_frames)

            # Add \textbf{} around the max value in a column (except first and second column) and put 3 dec
            for column in df_merged.columns[2:]:
                df_merged[column] = df_merged[column].apply(lambda x: f"\textbf{{{x*100:.1f}}}" if x == df_merged[column].max() else f"{x*100:.1f}")

            # Make sure directory exists
            os.makedirs(f'{save_folder}/{exp}', exist_ok=True)

            # Save the dataframe to a latex table and group by first column 
            df_merged.to_latex(f'{save_folder}/{exp}/{class_name}_{data_type}.tex', index=False, escape=False, multirow=True)

        # Set ticks to fontsize
        ax[j][i].tick_params(axis='both', which='major', labelsize=fontsize)

# Set the titles of the first row to the experiment names
for i, (exp, exp_data) in enumerate(experiments.items()):
    ax[0][i].set_title(exp_data["Name"], fontsize=fontsize)

# Set last column to rooms/icons (class_structure)
for j, (class_name, class_data) in enumerate(class_structure.items()):
    ax[j][-1].set_ylabel(class_name, fontsize=fontsize)
    ax[j][-1].yaxis.set_label_position("right")

# Get the labels and handles for the last
handles, labels = ax[-1][-1].get_legend_handles_labels()
unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]

# Plot legend outside of the plot on the right and for y placement between the rows
fig.legend(*zip(*unique), fontsize=fontsize, loc='center left', bbox_to_anchor=(1, 0.5))

# Add ylabel (Intersection over Union (IoU)) on the left side with vertical centering
fig.text(0.00, 0.5, "IoU (%)", va='center', rotation='vertical', fontsize=fontsize)

# Add xlabel (Experiment) on the bottom with horizontal centering
fig.text(0.5, 0.00, "Adaptive Pooling Size (HxW)", ha='center', fontsize=fontsize)

# Adjust the layout for the legend
plt.subplots_adjust(right=0.5)

# Save the figure
plt.tight_layout(pad=2.0)

# Make sure directory exists
plt.savefig(f'{save_folder}/MP_comparison.pdf', bbox_inches='tight')
            