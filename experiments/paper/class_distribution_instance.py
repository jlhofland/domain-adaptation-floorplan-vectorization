import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import matplotlib.cm as cm
from matplotlib import colors
import scipy.stats as stats

## DATA SETTINGS
data_folder = 'data'
save_folder = 'experiments\paper\plots\distribution'
files = [
    ['train_hqa_hq.txt', 'val_hqa_hq.txt', 'test_hqa_hq.txt'], # Source domain
    ['train_c.txt', 'val_c.txt', 'test_c.txt'], # Target domain
]
distribution_type = 'pixels' # 'classes' or 'pixels'

## LABELS ##
room_classes = ["No Room", "Outdoor", "Wall", "Kitchen", "Living", "Bedroom", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
icon_classes = ["No Icon", "Window", "Door", "Closet", "Elec. Appl.", "Toilet", "Sink", "Sauna", "Fireplace", "Bathtub", "Chimney"]
phases = ['Train', 'Validation', 'Test']
domains = ['Source', 'Target', 'Comparison']

nr_rooms = len(room_classes)
nr_icons = len(icon_classes)

## COLOR MAPS ##
cpool = ['#DCDCDC', '#b3de69', '#000000', '#8dd3c7', '#fdb462', # First was #DCDCDC
            '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969',
            '#577a4d', '#ffffb3']
cmap3 = colors.ListedColormap(cpool, 'rooms')
cm.register_cmap(cmap=cmap3)

cpool = ['#DCDCDC', '#8dd3c7', '#b15928', '#fdb462', '#ffff99',
            '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969',
            '#577a4d']
cmap3 = colors.ListedColormap(cpool, 'icons')
cm.register_cmap(cmap=cmap3)

room_colors = plt.cm.get_cmap('rooms')(np.linspace(0, 1, nr_rooms))
icon_colors = plt.cm.get_cmap('icons')(np.linspace(0, 1, nr_icons))

## PLOTTING ##
fig_rooms, ax_rooms = plt.subplots(len(files)+1, len(files[0])+1, figsize=(24, 12), gridspec_kw={'hspace': 0.5})
fig_icons, ax_icons = plt.subplots(len(files)+1, len(files[0])+1, figsize=(24, 12), gridspec_kw={'hspace': 0.5})

lines_styles = ['solid', 'dashed']
lines_colors = ['black', 'red', 'blue']

def get_unique(data_folder, file, dist_type='classes'):
    # Create empty lists
    rooms_list = []
    icons_list = []

    # Open the file
    with open(f'{data_folder}/{file}', 'r') as f:
        # Loop using tqdm to show progress bar
        for line in tqdm.tqdm(f):
            # Load data.pkl file
            data = np.load(f'{data_folder}/{line.strip()}/data.pkl', allow_pickle=True)

            label = data['label']
            rooms = label[0]
            icons = label[1]

            if dist_type == 'pixels':
                # Filter 0 values
                rooms = rooms[rooms != 0]
                icons = icons[icons != 0]
            elif dist_type == 'classes':
                # Unique values
                rooms = np.unique(rooms)
                icons = np.unique(icons)
            else:
                raise ValueError(f'Unknown dist_type {dist_type}')

            # Add the rooms and icons to the list
            rooms_list.append(rooms)
            icons_list.append(icons)

            if len(rooms_list) == 2000:
                break

    return np.concatenate(rooms_list), np.concatenate(icons_list)

domain_densities = {}

def kl_divergence_phases(train, validation, test):
    kl_train_val = stats.entropy(train, validation)
    kl_train_test = stats.entropy(train, test)
    kl_val_test = stats.entropy(validation, test)

    return np.mean([kl_train_val, kl_train_test, kl_val_test])

for i, domain in enumerate(files):
    densities = {"rooms": [], "icons": []}
    for j, phase in enumerate(domain):
        # Get the unique rooms and icons
        rooms, icons = get_unique(data_folder, phase, dist_type=distribution_type)
        
        ## LABELS ##
        if (i == 0):
            ax_rooms[i, j].set_title(phases[j])
            ax_icons[i, j].set_title(phases[j])

        if (j == 0):
            ax_rooms[i, j].set_ylabel(domains[i])
            ax_icons[i, j].set_ylabel(domains[i])

        ## ROOMS ## 
        ax_rooms[i, j].hist(rooms, bins=nr_rooms, range=(0, nr_rooms), align='left', rwidth=0.8, density=True)
        for k in range(nr_rooms):
            ax_rooms[i, j].text(k, ax_rooms[i, j].patches[k].get_height(), f'{ax_rooms[i, j].patches[k].get_height() * 100:.1f}', ha='center', va='bottom')

        for b, bar in enumerate(ax_rooms[i, j].patches):
            bar.set_facecolor(room_colors[b])

        # Remove border around subplot and set x-axis to empty
        ax_rooms[i, j].spines['top'].set_visible(False)
        ax_rooms[i, j].spines['right'].set_visible(False)
        ax_rooms[i, j].set_xticks([])\
        
        print(f'{domain} {phase} rooms density calculated')

        # Get the kernel density estimate
        kernel = stats.gaussian_kde(rooms)
        x = np.linspace(0, nr_rooms, 200)
        y = kernel(x)
        densities["rooms"].append(y)

        # Plot in last col of row and last row of col
        ax_rooms[i, len(domain)].plot(x, y, color=lines_colors[j], linestyle=lines_styles[i])
        ax_rooms[len(files), j].plot(x, y, color=lines_colors[j], linestyle=lines_styles[i])

        # set x-ticks and labels
        ax_rooms[i, j].set_xticks(range(nr_rooms))
        ax_rooms[i, j].set_xticklabels(room_classes, rotation=45)

        print(f'{domain} {phase} rooms kernel density calculated')

        ## ICONS ##
        ax_icons[i, j].hist(icons, bins=nr_icons, range=(0, nr_icons), align='left', rwidth=0.8, density=True)
        for k in range(nr_icons):
            ax_icons[i, j].text(k, ax_icons[i, j].patches[k].get_height(), f'{ax_icons[i, j].patches[k].get_height() * 100:.1f}', ha='center', va='bottom')

        for b, bar in enumerate(ax_icons[i, j].patches):
            bar.set_facecolor(icon_colors[b])

        # Remove border around subplot and set x-axis to empty
        ax_icons[i, j].spines['top'].set_visible(False)
        ax_icons[i, j].spines['right'].set_visible(False)
        ax_icons[i, j].set_xticks([])

        print(f'{domain} {phase} icons density calculated')

        # Get the kernel density estimate
        kernel = stats.gaussian_kde(icons)
        x = np.linspace(0, nr_icons, 200)
        y = kernel(x)
        densities["icons"].append(y)

        # Plot in last col of row and last row of col
        ax_icons[i, len(domain)].plot(x, y, color=lines_colors[j], linestyle=lines_styles[i])
        ax_icons[len(files), j].plot(x, y, color=lines_colors[j], linestyle=lines_styles[i])

        # set x-ticks and labels
        ax_icons[i, j].set_xticks(range(nr_icons))
        ax_icons[i, j].set_xticklabels(icon_classes, rotation=45)

        print(f'{domain} {phase} icons kernel density calculated')

    # Plot the mean of the densities and add the legend
    ax_rooms[len(files), len(domain)].plot(x, np.mean(densities["rooms"], axis=0), color='black', linestyle=lines_styles[i])
    ax_icons[len(files), len(domain)].plot(x, np.mean(densities["icons"], axis=0), color='black', linestyle=lines_styles[i])

    # KL-Divergence between phases
    kl_rooms = kl_divergence_phases(densities["rooms"][0], densities["rooms"][1], densities["rooms"][2])
    kl_icons = kl_divergence_phases(densities["icons"][0], densities["icons"][1], densities["icons"][2])

    # Plot the KL div at lower left corner
    ax_rooms[i, len(domain)].text(0.1, 0.1, f'KL-divergence: {kl_rooms:.4f}', transform=ax_rooms[i, len(domain)].transAxes)
    ax_icons[i, len(domain)].text(0.1, 0.1, f'KL-divergence: {kl_icons:.4f}', transform=ax_icons[i, len(domain)].transAxes)

    # Save the densities
    domain_densities[domains[i]] = densities

    print(f'{domain} finished')

# CLASS LABELS AND LEGENDS
for i in range(len(files[0])):
    # ROOMS
    ax_rooms[len(files), i].set_xticks(range(nr_rooms))
    ax_rooms[len(files), i].set_xticklabels(room_classes, rotation=45)
    ax_rooms[len(files), i].legend(domains[:2], loc='upper right')
    ax_rooms[len(files), i].spines['top'].set_visible(False)
    ax_rooms[len(files), i].spines['right'].set_visible(False)

    kl_rooms = stats.entropy(domain_densities['Source']['rooms'][i], domain_densities['Target']['rooms'][i])
    ax_rooms[len(files), i].text(0.1, 0.1, f'KL-divergence: {kl_rooms:.4f}', transform=ax_rooms[len(files), i].transAxes)

    kl_icons = stats.entropy(domain_densities['Source']['icons'][i], domain_densities['Target']['icons'][i])
    ax_icons[len(files), i].text(0.1, 0.1, f'KL-divergence: {kl_icons:.4f}', transform=ax_icons[len(files), i].transAxes)

    # ICONS
    ax_icons[len(files), i].set_xticks(range(nr_icons))
    ax_icons[len(files), i].set_xticklabels(icon_classes, rotation=45)
    ax_icons[len(files), i].legend(domains[:2], loc='upper right')
    ax_icons[len(files), i].spines['top'].set_visible(False)
    ax_icons[len(files), i].spines['right'].set_visible(False)

for j in range(len(files)):
    # ROOMS
    ax_rooms[j, len(files[0])].set_xticks(range(nr_rooms))
    ax_rooms[j, len(files[0])].set_xticklabels(room_classes, rotation=45)
    ax_rooms[j, len(files[0])].legend(phases, loc='upper right')
    ax_rooms[j, len(files[0])].spines['top'].set_visible(False)
    ax_rooms[j, len(files[0])].spines['right'].set_visible(False)

    # ICONS
    ax_icons[j, len(files[0])].set_xticks(range(nr_icons))
    ax_icons[j, len(files[0])].set_xticklabels(icon_classes, rotation=45)
    ax_icons[j, len(files[0])].legend(phases, loc='upper right')
    ax_icons[j, len(files[0])].spines['top'].set_visible(False)
    ax_icons[j, len(files[0])].spines['right'].set_visible(False)

# SET Comparison label for last row
ax_rooms[len(files), 0].set_ylabel(domains[2])
ax_rooms[0, len(files[0])].set_title(domains[2])

ax_icons[len(files), 0].set_ylabel(domains[2])
ax_icons[0, len(files[0])].set_title(domains[2])

## LAST CELL (COMPARISON)
ax_rooms[len(files), len(domain)].legend(domains[:2], loc='upper right')
ax_icons[len(files), len(domain)].legend(domains[:2], loc='upper right')

kl_rooms = stats.entropy(densities["rooms"][0], densities["rooms"][1])
kl_icons = stats.entropy(densities["icons"][0], densities["icons"][1])

# Plot the KL div at lower left corner
ax_rooms[len(files), len(domain)].text(0.1, 0.1, f'KL-divergence: {kl_rooms:.4f}', transform=ax_rooms[len(files), len(domain)].transAxes)
ax_icons[len(files), len(domain)].text(0.1, 0.1, f'KL-divergence: {kl_icons:.4f}', transform=ax_icons[len(files), len(domain)].transAxes)

ax_rooms[len(files), len(domain)].set_xticks(range(nr_rooms))
ax_rooms[len(files), len(domain)].set_xticklabels(room_classes, rotation=45)

ax_icons[len(files), len(domain)].set_xticks(range(nr_icons))
ax_icons[len(files), len(domain)].set_xticklabels(icon_classes, rotation=45)

ax_rooms[len(files), len(domain)].spines['top'].set_visible(False)
ax_rooms[len(files), len(domain)].spines['right'].set_visible(False)

ax_icons[len(files), len(domain)].spines['top'].set_visible(False)
ax_icons[len(files), len(domain)].spines['right'].set_visible(False)

# Save the figures
fig_rooms.savefig(f'{save_folder}/class_{distribution_type}_distribution_rooms.png')
fig_icons.savefig(f'{save_folder}/class_{distribution_type}_distribution_icons.png')

