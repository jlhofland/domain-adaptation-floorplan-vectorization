import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# image files
# domain = {"High Quality": "test_hq.txt", "Colored": "test_c.txt", "High Quality Architectural": "test_hqa.txt"}
domain = {"Source": "test_hqa_hq.txt", "Target": "test_c.txt" }
folder = "data/"
save_folder = 'experiments\paper\plots\distribution'
legend = []
channels = 3
domain_densities = []
apply_mask = "all"

# Create figure with subplots of height 3 and width 1
fig, ax = plt.subplots(len(domain)+1, channels+1, figsize=(24, 6), gridspec_kw={'hspace': 0.7, 'wspace': 0.2})

def consine_similarity(r, g, b):
    rg = np.dot(r, g) / (np.linalg.norm(r) * np.linalg.norm(g))
    rb = np.dot(r, b) / (np.linalg.norm(r) * np.linalg.norm(b))
    gb = np.dot(g, b) / (np.linalg.norm(g) * np.linalg.norm(b))
    return np.mean([rg, rb, gb])

def cosine_similarity_2(d1, d2):
    return np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))

def kl_divergence(r, g, b):
    rg = kl_divergence_2(r, g)
    rb = kl_divergence_2(r, b)
    gb = kl_divergence_2(g, b)
    return np.mean([rg, rb, gb])

def kl_divergence_2(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

# Load the data
for i, (k, v) in enumerate(domain.items()):
    pixels_dict  = {'Red': [], 'Green': [], 'Blue': []}
    density_dict = {'Red': [], 'Green': [], 'Blue': []}

    # Open the file
    with open(f'{folder}/{v}', 'r') as f:
        # Loop using tqdm to show progress bar
        for line in tqdm.tqdm(f):
            # Load data.pkl file
            data = np.load(f'{folder}/{line.strip()}/data.pkl', allow_pickle=True)

            # Get the image
            image = data['image'] # Shape: (3, H, W)
            label = data['label'] 

            # Get the labels
            rooms = label[0]
            icons = label[1]

            # Append the pixels to each channel list
            for j, channel in enumerate(pixels_dict.keys()):
                if apply_mask == "icons":
                    pixels_dict[channel].append(image[j, :, :][icons != 0].flatten()) # Only icon pixels
                elif apply_mask == "rooms":
                    pixels_dict[channel].append(image[j, :, :][(rooms != 0) & (icons == 0)].flatten()) # Only room pixels
                else: 
                    pixels_dict[channel].append(image[j, :, :].flatten()) # All pixels

            # if len(pixels_dict['Red']) == 3:
            #     break

        # Each domain has its own line type
        line_type = 'solid' if i == 0 else 'dashed'

        # Plot the histogram
        for j, channel in enumerate(pixels_dict.keys()):
            # Concatenate the pixels
            pixels_dict[channel] = np.concatenate(pixels_dict[channel])

            # Calculate the mean and standard deviation
            mean = np.mean(pixels_dict[channel])
            std = np.std(pixels_dict[channel])

            # Create histogram of pixels with the count of each pixel above the bars with log scale
            ax[i, j].hist(pixels_dict[channel], bins=256, rwidth=0.8, density=True, color=channel.lower())
            ax[i, j].axvline(mean, color=channel.lower(), linestyle='dashed', linewidth=1)
            ax[i, j].axvline(mean - std, color='k', linestyle='dashed', linewidth=1)

            # Set to log scale
            ax[i, j].set_yscale('log')

            # Add the mean and standard deviation to the plot (top left corner)
            ax[i, j].text(0.05, 0.95, f'Mean: {mean:.2f}\nStd: {std:.2f}', transform=ax[i, j].transAxes, fontsize=8, verticalalignment='top')

            print(f'{k} {channel} mean: {mean:.2f} std: {std:.2f}')

            # Get the kernel density estimate
            kernel = stats.gaussian_kde(pixels_dict[channel])

            print(f'{k} {channel} kernel density estimated')

            # Plot the kernel density estimate
            x = np.linspace(min(pixels_dict[channel]), max(pixels_dict[channel]), 256)

            # Get the density values
            y = kernel(x)

            print(f'{k} {channel} kernel density calculated')

            # Add the density values to the dictionary
            density_dict[channel] = y

            # Plot the density values
            ax[len(domain), j].plot(x, y, color=channel.lower(), linestyle=line_type)

            # Set scale to log
            ax[len(domain), j].set_yscale('log')

            # Set the title
            ax[len(domain), j].set_xlabel('Pixel intensity')

            # Plot the density values in the last col
            ax[i, channels].plot(x, y, color=channel.lower(), linestyle=line_type)

            # Set scale to log
            ax[i, channels].set_yscale('log')

            print(f'{k} {channel} kernel density plotted')

    # Calculate the similarity between the domains (pearson correlation, cosine similarity, jacard similarity)
    # similarity_per = pearson_correlation(density_dict['Red'], density_dict['Green'], density_dict['Blue'])
    similarity_cos = consine_similarity(density_dict['Red'], density_dict['Green'], density_dict['Blue'])
    similarity_kld = kl_divergence(density_dict['Red'], density_dict['Green'], density_dict['Blue'])

    # Add the both the similarity and cosine similarity to the plot
    ax[i, channels].text(0.05, 0.95, f'Cosine similarity: {similarity_cos:.2f}\nKL-divergence: {similarity_kld:.2f}', transform=ax[i, channels].transAxes, fontsize=8, verticalalignment='top')

    # Add the density values to the domain densities
    domain_densities.append(density_dict)

    # Plot the average density values in the last cell
    ax[len(domain), channels].plot(x, np.mean([density_dict['Red'], density_dict['Green'], density_dict['Blue']], axis=0), color='k', linestyle=line_type)

    # Add legend
    legend.append(k)

# Add the name of the domains to the left column
for i, d in enumerate(domain.keys()):
    ax[i, 0].set_ylabel(d)

# Add the name of the channels to the top row
ax[len(domain), 0].set_ylabel("Comparison")

# Calculate the similarity between the domains for each channel
for j, channel in enumerate(pixels_dict.keys()):
    # Calculate the similarity between the domains (pearson correlation, cosine similarity, jacard similarity)
    # similarity_per = pearson_correlation_2(domain_densities[0][channel], domain_densities[1][channel])
    similarity_cos = cosine_similarity_2(domain_densities[0][channel], domain_densities[1][channel])
    similarity_kld = kl_divergence_2(domain_densities[0][channel], domain_densities[1][channel])

    # Add the both the similarity and cosine similarity to the plot
    ax[len(domain), j].text(0.05, 0.95, f'Cosine similarity: {similarity_cos:.2f}\nKL-divergence: {similarity_kld:.2f}', transform=ax[len(domain), j].transAxes, fontsize=8, verticalalignment='top')

# Calculate the similarity between the domains for mean of each channel
similarity_cos = cosine_similarity_2(np.mean([domain_densities[0]['Red'], domain_densities[0]['Green'], domain_densities[0]['Blue']], axis=0), np.mean([domain_densities[1]['Red'], domain_densities[1]['Green'], domain_densities[1]['Blue']], axis=0))
similarity_kld = kl_divergence_2(np.mean([domain_densities[0]['Red'], domain_densities[0]['Green'], domain_densities[0]['Blue']], axis=0), np.mean([domain_densities[1]['Red'], domain_densities[1]['Green'], domain_densities[1]['Blue']], axis=0))

# Add to the plot
ax[len(domain), channels].text(0.05, 0.95, f'Cosine similarity: {similarity_cos:.2f}\nKL-divergence: {similarity_kld:.2f}', transform=ax[len(domain), channels].transAxes, fontsize=8, verticalalignment='top')

# Plot legend in the bottom right corner
ax[len(domain), channels].legend(legend, loc='lower right', fontsize=8)
ax[len(domain), 0].legend(legend, loc='lower right', fontsize=8)
ax[len(domain), 1].legend(legend, loc='lower right', fontsize=8)
ax[len(domain), 2].legend(legend, loc='lower right', fontsize=8)


# Add axis to the last cell
ax[len(domain), channels].set_yscale('log')
ax[len(domain), channels].set_xlabel('Pixel intensity')

# Save the plot to a file
plt.savefig(f'{save_folder}/pixel_distribution_{apply_mask}.png')
