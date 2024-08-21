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

style = {
    "Red": "solid",
    "Green": "dashed",
    "Blue": "dotted"
}
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
fig, ax = plt.subplots(len(structure), len(structure["Colored"]), figsize=(len(structure["Colored"])*8, len(structure)*4), sharex='col', sharey='row', gridspec_kw = {'wspace':0, 'hspace':0})

domain_dict = {}
for i, (domain, domain_data) in enumerate(structure.items()):
    data_dict = {}
    for j, (data_type, data_file) in enumerate(domain_data.items()):
        pixels = {"Red": [], "Green": [], "Blue": []}
        with open(f'{data_folder}/{data_file}', 'r') as f:
            # Loop using tqdm to show progress bar
            for line in tqdm.tqdm(f):
                # Load data.pkl file
                data = np.load(f'{data_folder}/{line.strip()}/data.pkl', allow_pickle=True)

                # Get image
                image = data['image']

                # Loop through the channels
                for k, channel in enumerate(["Red", "Green", "Blue"]):
                    pixels[channel].append(image[k, :, :].flatten())

                # if len(pixels["Red"]) == 100:
                #     break

        # Get the number of samples
        n_samples = len(pixels["Red"])

        # Add grid
        ax[i, j].grid(True, alpha=0.5)

        # For each channel convert to numpy array
        for k, channel in enumerate(["Red", "Green", "Blue"]):
            pixels[channel] = np.histogram(np.concatenate(pixels[channel]), bins=int(256), range=(0, 256), density=True)

            # Plot a line plot
            ax[i, j].plot(pixels[channel][1][:-1], pixels[channel][0]*100, label=channel, color=channel, alpha=0.7, linestyle=style[channel])

        # Calculate KL divergence
        kl = calculate_average_kl([pixels["Red"][0], pixels["Green"][0], pixels["Blue"][0]])

        # Plot kl diverge upper left and n_samples
        ax[i, j].text(0.05, 0.95, f"KL: {kl:.2f}\nN: {n_samples}", transform=ax[i, j].transAxes, fontsize=fontsize, verticalalignment='top')

        # Put log scale on y-axis
        ax[i, j].set_yscale('log')

        # Show y-axis from 10-5 to 10-2
        # ax[i, j].set_ylim(100*10**-4, 100*10**-1)

        # Set ylabels to normal percentage format
        ax[i, j].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    
        # Set tick to fontsize
        ax[i, j].tick_params(axis='both', which='major', labelsize=fontsize)

        # Add the pixel denstiy to dict
        data_dict[data_type] = pixels

    kl = 0

    # For all colors
    for color in ["Red", "Green", "Blue"]:
        # Calculate KL divergence
        kl += calculate_average_kl([data_dict["Train"][color][0], data_dict["Validation"][color][0], data_dict["Test"][color][0]])

    # Add axis titles/labels + kl divergence
    ax[i, -1].set_ylabel(f"{domain}\nKL: {kl/3:.2f}", fontsize=fontsize)
    ax[i, -1].yaxis.set_label_position("right")

    # Add in dict
    domain_dict[domain] = data_dict

# Set ylabel of figure
ax[1, 0].set_ylabel("Density (%)", fontsize=fontsize)
ax[-1, 1].set_xlabel("Pixel Intensity", fontsize=fontsize)

# Handle special rows
for j, (data_type, data_file) in enumerate(domain_data.items()):
    # For all colors calculate KL divergence
    kl = 0
    for color in ["Red", "Green", "Blue"]:
        kl += calculate_average_kl([domain_dict["High Quality Architectural"][data_type][color][0], domain_dict["High Quality"][data_type][color][0], domain_dict["Colored"][data_type][color][0]])
    
    # Set first row title to data type
    ax[0, j].set_title(f"{data_type}\nKL: {kl/3:.2f}", fontsize=fontsize)

# Create a 3x3 dataframe (3 domains, 3 domains) and calculate the KL divergence between the domains (asymmetric)
kl_df = pd.DataFrame(index=structure.keys(), columns=structure.keys())
for i, (domain1, domain1_data) in enumerate(domain_dict.items()):
    for j, (domain2, domain2_data) in enumerate(domain_dict.items()):
        kl = []
        for color in ["Red", "Green", "Blue"]:
            domain1_app = domain1_data["Train"][color][0] + domain1_data["Validation"][color][0] + domain1_data["Test"][color][0],
            domain2_app = domain2_data["Train"][color][0] + domain2_data["Validation"][color][0] + domain2_data["Test"][color][0]

            # Normalize the data
            domain1_app = domain1_app / np.sum(domain1_app)
            domain2_app = domain2_app / np.sum(domain2_app)

            kl.append(kl_divergence(domain1_app, domain2_app))
        kl_df.loc[domain1, domain2] = f'{np.mean(kl):.2f}'

        print(f"KL divergence from {domain1} to {domain2}: {np.mean(kl):.2f}")

# Save the dataframe to a latex table
kl_df.to_latex(f'{save_folder}/pixel_distribution_per_domain_split.tex', escape=False)

# Save the plot
plt.tight_layout()
plt.savefig(f'{save_folder}/pixel_distribution_per_domain_split.pdf', bbox_inches='tight')




                


