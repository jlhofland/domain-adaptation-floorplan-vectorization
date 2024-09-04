import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_epochs = 100
num_samples = 100
img_size = 64
num_classes = 3

# Simulate synthetic data (random values for illustration)
heatmaps = np.random.rand(num_samples, img_size, img_size)
rooms = np.random.randint(0, num_classes, (num_samples, img_size, img_size))
icons = np.random.randint(0, num_classes, (num_samples, img_size, img_size))

# Simulate losses, uncertainties, and weights over epochs
epochs = np.arange(1, num_epochs + 1)
loss_heatmap = np.exp(-0.01 * epochs)  # Simulated decreasing loss
loss_rooms = np.exp(-0.015 * epochs)   # Simulated decreasing loss
loss_icons = np.exp(-0.02 * epochs)    # Simulated decreasing loss

sigma_heatmap = 0.1 + 0.01 * epochs    # Simulated increasing uncertainty
sigma_rooms = 0.2 + 0.01 * epochs      # Simulated increasing uncertainty
sigma_icons = 0.3 + 0.01 * epochs      # Simulated increasing uncertainty

weight_heatmap = 1 / (sigma_heatmap ** 2)
weight_rooms = 1 / (sigma_rooms ** 2)
weight_icons = 1 / (sigma_icons ** 2)

# Total loss (weighted sum)
total_loss = (weight_heatmap * loss_heatmap + weight_rooms * loss_rooms + weight_icons * loss_icons)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Task Loss Contributions Over Time
axs[0].plot(epochs, loss_heatmap, label='Heatmap Loss')
axs[0].plot(epochs, loss_rooms, label='Rooms Loss')
axs[0].plot(epochs, loss_icons, label='Icons Loss')
axs[0].plot(epochs, total_loss, label='Total Loss', linestyle='--')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].set_title('Task Loss Contributions Over Time')

# Plot 2: Uncertainty and Weight Evolution
color = 'tab:blue'
axs2 = axs[1].twinx()
axs[1].plot(epochs, sigma_heatmap, label='Heatmap Sigma', color=color)
axs[1].plot(epochs, sigma_rooms, label='Rooms Sigma', linestyle='--', color=color)
axs[1].plot(epochs, sigma_icons, label='Icons Sigma', linestyle='-.', color=color)
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Sigma', color=color)
axs[1].tick_params(axis='y', labelcolor=color)
axs[1].legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

color = 'tab:red'
axs2.plot(epochs, weight_heatmap, label='Heatmap Weight', color=color)
axs2.plot(epochs, weight_rooms, label='Rooms Weight', linestyle='--', color=color)
axs2.plot(epochs, weight_icons, label='Icons Weight', linestyle='-.', color=color)
axs2.set_ylabel('Weight', color=color)
axs2.tick_params(axis='y', labelcolor=color)
axs2.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

axs[1].set_title('Uncertainty and Weight Evolution Over Time')

fig.tight_layout()
plt.show()
