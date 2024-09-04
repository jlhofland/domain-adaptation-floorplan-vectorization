import torch
from omegaconf import OmegaConf
from datasets.cubicasa5k.loaders.svg_loader import FloorplanSVG
from model_factory import factory
import matplotlib.pyplot as plt
import tqdm
from torchvision.transforms import Compose
from datasets.cubicasa5k.loaders.augmentations import DictToTensor
import numpy as np

from datasets.cubicasa5k.metrics import CustomMetric, polygons_to_tensor
from datasets.cubicasa5k.post_prosessing import get_polygons

import torch.nn.functional as F
from datasets.cubicasa5k.loaders.augmentations import RotateNTurns

from datasets.cubicasa5k.plotting import discrete_cmap

import os
import pandas as pd

if __name__ == "__main__":
    discrete_cmap()

    example_folder = 'paper/plots/PM'
    os.makedirs(example_folder, exist_ok=True)

    max_count = 1

    colors = {
        "Rooms": ['#DCDCDC', '#b3de69', '#000000', '#8dd3c7', '#fdb462', '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969', '#577a4d', '#ffffb3'],
        "Icons": ['#DCDCDC', '#8dd3c7', '#b15928', '#fdb462', '#ffff99', '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969', '#577a4d']
    }

    labels = {
        "Rooms": ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"],
        "Icons": ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]
    }

    fontsize = 18
    model_weights = "weights/model_best_val_loss_var.pkl"

    # Load defaults and overwrite by command-line arguments
    cfg = OmegaConf.load("exp_archive/config_pkl.yaml")
    cmd_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cmd_cfg)
    print(OmegaConf.to_yaml(cfg))

    dataset = FloorplanSVG('test_hqa.txt', cfg, pre_load=True, augmentations=Compose([DictToTensor()]))
    loader  = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    # Load model weights
    weights = torch.load(model_weights, map_location=torch.device('cpu'))

    # Create model
    model = factory(cfg)

    # Load model weights
    model.load_state_dict(weights['model_state'])

    # Set model to evaluation mode
    model.eval()

    # Loop over data
    with torch.no_grad():
        for count, val in tqdm.tqdm(enumerate(loader), total=len(loader), ncols=80, leave=False):
            # Unpack data
            image = val['image']
            label = val['label']

            # Img size
            batch, _, height, width = image.shape

            # Forward pass
            prediction, _ = model(image)

            # Interpolate the prediction to the original size and move to CPU
            prediction = F.interpolate(prediction, size=(height, width), mode='bilinear', align_corners=False) 

            # Split the tensor into heatmaps, rooms and icons
            heats, rooms, icons = torch.split(prediction, tuple(cfg.model.input_slice), 1)

            # Get the rooms and icons
            icons = F.softmax(icons, dim=1)
            rooms = F.softmax(rooms, dim=1)

            # Transform label 
            label = label[:, cfg.model.input_slice[0]:][0] # Rooms and Icons

            # Remove batch dimension
            image = image[0] 
            rooms = rooms[0] 
            icons = icons[0]
            heats = heats[0]
            label = label[0]

            # Normalize input image for imshow
            image = (image - image.min()) / (image.max() - image.min())

            num_classes = 10
            alpha = 0.5
            cmap = 'viridis' # 'viridis'

            # Visualize the first 3 probabilities maps of rooms and icons and heatmaps
            fig, axs = plt.subplots(3, num_classes, figsize=(5*num_classes, 15))

            # Plot the heatmaps for rooms
            for i, (ax, room) in enumerate(zip(axs[0], rooms)):
                ax.imshow(image.permute(1, 2, 0))
                ax.imshow(room, alpha=alpha, cmap=cmap)
                ax.set_title(labels["Rooms"][i], fontsize=fontsize)
                ax.axis('off')

            # Plot the heatmaps for icons
            for i, (ax, icon) in enumerate(zip(axs[1], icons)):
                ax.imshow(image.permute(1, 2, 0))
                ax.imshow(icon, alpha=alpha, cmap=cmap)
                ax.set_title(labels["Icons"][i], fontsize=fontsize)
                ax.axis('off')

            # Plot the heatmaps for heatmaps
            for i, (ax, heat) in enumerate(zip(axs[2], heats)):
                ax.imshow(image.permute(1, 2, 0))
                ax.imshow(heat, alpha=alpha, cmap=cmap)
                ax.set_title(f"Heatmap {i}", fontsize=fontsize)
                ax.axis('off')

            # Save the plot
            plt.tight_layout()
            plt.savefig(f"probability_example_{count}.png")

            if count == max_count:
                break
