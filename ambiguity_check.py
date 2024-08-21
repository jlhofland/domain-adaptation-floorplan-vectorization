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

    structural_folder = 'paper/plots/AC/structural'
    undefined_folder = 'paper/plots/AC/undefined'
    os.makedirs(structural_folder, exist_ok=True)
    os.makedirs(undefined_folder, exist_ok=True)

    max_count = -1

    colors = {
        "Rooms": ['#DCDCDC', '#b3de69', '#000000', '#8dd3c7', '#fdb462', '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969', '#577a4d', '#ffffb3'],
        "Icons": ['#DCDCDC', '#8dd3c7', '#b15928', '#fdb462', '#ffff99', '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969', '#577a4d']
    }

    files = {
        "Colored": {
            "Train": {
                "file": "train_c.txt",
                "ignore": []
            },
            "Val": {
                "file": "val_c.txt",
                "ignore": [43]
            },
            "Test": {
                "file": "test_c.txt",
                "ignore": [19]
            }
        },
        "High Quality": {
            "Train": {
                "file": "train_hq.txt",
                "ignore": [0, 12, 14, 65, 154, 322, 356, 571, 576, 654, 717, 807]
            },
            "Val": {
                "file": "val_hq.txt",
                "ignore": []
            },
            "Test": {
                "file": "test_hq.txt",
                "ignore": [4]
            }
        },
        "High Quality Architectural": {
            "Train": {
                "file": "train_hqa.txt",
                "ignore": [207, 334, 1053, 1060, 1263, 1457, 1562, 1660, 1768, 1812, 1906, 1920, 1921, 1922, 1923, 1948, 2386, 2501, 2505, 2702, 2759]
            },
            "Val": {
                "file": "val_hqa.txt",
                "ignore": [60]
            },
            "Test": {
                "file": "test_hqa.txt",
                "ignore": [37, 84]
            }
        },
    }
    labels = {
        "Rooms": ["Background", "Outdoor", "Wall", "Kitchen", "Living \nRoom" ,"Bed \nRoom", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"],
        "Icons": ["No Icon", "Window", "Door", "Closet", "Electrical \nApplience" ,"Toilet", "Sink", "Sauna \nBench", "Fire Place", "Bathtub", "Chimney"]
    }
    fontsize = 18
    model_weights = "weights/model_best_val_loss_var.pkl"

    # Load defaults and overwrite by command-line arguments
    cfg = OmegaConf.load("config_pkl.yaml")
    cmd_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cmd_cfg)
    print(OmegaConf.to_yaml(cfg))

    domain_scores = {}

    if torch.backends.mps.is_available():
        print("MPS available and enabled")

    # Print number of GPUS and cpu threads
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Number of CPU threads: {torch.get_num_threads()}")

    # Load model weights
    weights = torch.load(model_weights, map_location=torch.device('cpu'))

    # Create model
    model = factory(cfg)

    # Load model weights
    model.load_state_dict(weights['model_state'])

    # Set model to evaluation mode
    model.eval()

    for i, (domain, splits) in enumerate(files.items()):
        domain_scores[domain] = {}
        for j, (split, split_data) in enumerate(splits.items()):
            # Load data
            dataset = FloorplanSVG(split_data["file"], cfg, pre_load=True, augmentations=Compose([DictToTensor()]))
            loader  = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)

            # Add domain and split to dictionary
            domain_scores[domain][split] = {"undefined": [], "structure": [], "total_len": len(loader) if max_count == -1 else max_count}

            # Loop over data
            with torch.no_grad():
                for count, val in tqdm.tqdm(enumerate(loader), total=len(loader), desc=f"Domain: {domain}", ncols=80, leave=False):
                    # Unpack data
                    image = val['image']
                    label = val['label']

                    # Img size
                    batch, _, height, width = image.shape

                    # # Create rotations
                    # rot_class = RotateNTurns()
                    # rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]

                    # # Create prediction tensor
                    # pred_count = len(rotations)
                    # prediction = torch.zeros([pred_count, sum(cfg.model.input_slice), height, width])

                    # # Loop over rotations
                    # for i, r in enumerate(rotations):
                    #     forward, back = r
                    #     # We rotate first the image
                    #     rot_image = rot_class(image, 'tensor', forward)
                    #     pred, latent = model(rot_image)

                    #     # We rotate prediction back
                    #     pred = rot_class(pred, 'tensor', back)

                    #     # We fix heatmaps
                    #     pred = rot_class(pred, 'points', back)

                    #     # We make sure the size is correct
                    #     pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)

                    #     # We add the prediction to output
                    #     prediction[i] = pred[0]

                    # # We average the predictions
                    # prediction = torch.mean(prediction, dim=0, keepdim=True)

                    # Forward pass
                    prediction, _ = model(image)

                    # Interpolate the prediction to the original size and move to CPU
                    prediction = F.interpolate(prediction, size=(height, width), mode='bilinear', align_corners=False) 

                    # Split the tensor into heatmaps, rooms and icons
                    heats, rooms, icons = torch.split(prediction, tuple(cfg.model.input_slice), 1)

                    # Take softmax of the rooms and icons
                    icons = torch.argmax(F.softmax(icons, dim=1), dim=1)
                    rooms = torch.argmax(F.softmax(rooms, dim=1), dim=1)

                    # Transform label 
                    label = label[:, cfg.model.input_slice[0]:] # Rooms and Icons

                    # Remove batch dimension
                    image = image[0] 
                    rooms = rooms[0] 
                    label = label[0]

                    # Create a mask for the structure
                    s_mask = torch.zeros_like(rooms)
                    s_mask[(label[0] == 0) & (rooms != 0)] = 1

                    # Calculate percentage of structure
                    s_percentage = torch.sum(s_mask).item() / torch.sum(label[0] == 0).item()
                    st_percentage = torch.sum(s_mask).item() / len(s_mask.flatten())

                    # Undefinied mask 
                    u_mask = torch.zeros_like(rooms)
                    u_mask[(rooms != 0) & (label[0] == 11) & (rooms != 11)] = 1

                    # Calculate percentage of undefined
                    u_percentage  = torch.sum(u_mask).item() / torch.sum(rooms != 0).item()
                    ul_percentage = torch.sum(label[0] == 11).item() / torch.sum(label[0] != 0).item()

                    # Normalize input image for imshow
                    image = (image - image.min()) / (image.max() - image.min())

                    # Check if the percentage is above a certain threshold, also check if mask takes a significant part of the image
                    if s_percentage > 0.3 and st_percentage > 0.05 and count not in split_data["ignore"]:
                        domain_scores[domain][split]["structure"].append({
                            "idx": count,
                            "div": s_percentage
                        })

                        # Plot the image
                        fig, ax = plt.subplots(1, 4, figsize=(24, 8), sharey=True)
                        ax[0].imshow(image.permute(1, 2, 0))
                        ax[1].imshow(rooms, cmap='rooms', vmin=0, vmax=11-0.1) # possibly put vmax=19
                        ax[2].imshow(label[0], cmap='rooms', vmin=0, vmax=11-0.1)
                        ax[3].imshow(rooms * s_mask, cmap='rooms', vmin=0, vmax=11-0.1)

                        # Remove ticks from all axes (and labels)
                        for a in ax:
                            a.set_xticks([])
                            a.set_yticks([])
                            a.set_xticklabels([])
                            a.set_yticklabels([])

                        # Set titles
                        ax[0].set_title("Image", fontsize=fontsize)
                        ax[1].set_title("Prediction", fontsize=fontsize)
                        ax[2].set_title("Label", fontsize=fontsize)
                        ax[3].set_title(f"Error: {s_percentage:.2f}", fontsize=fontsize)

                        # Set the colorbar
                        plt.tight_layout()
                        
                        # Save the plot
                        plt.savefig(f'{structural_folder}/{domain}_{split}_{count}.pdf', bbox_inches='tight')

                    # Check if the percentage is above a certain threshold, also check if most of the image is undefined, also check if the image is not in the ignore list
                    if u_percentage > 0.3 and ul_percentage > 0.5: 
                        domain_scores[domain][split]["undefined"].append({
                            "idx": count,
                            "div": u_percentage
                        })

                        # Plot the image
                        fig, ax = plt.subplots(1, 4, figsize=(24, 8), sharey=True)
                        ax[0].imshow(image.permute(1, 2, 0))
                        ax[1].imshow(rooms, cmap='rooms', vmin=0, vmax=11-.1)
                        ax[2].imshow(label[0], cmap='rooms', vmin=0, vmax=11-.1)
                        ax[3].imshow(rooms * u_mask, cmap='rooms', vmin=0, vmax=11-.1)

                        # Remove ticks from all axes (and labels)
                        for a in ax:
                            a.set_xticks([])
                            a.set_yticks([])
                            a.set_xticklabels([])
                            a.set_yticklabels([])

                        # Set titles
                        ax[0].set_title("Image", fontsize=fontsize)
                        ax[1].set_title("Prediction", fontsize=fontsize)
                        ax[2].set_title("Label", fontsize=fontsize)
                        ax[3].set_title(f"Error: {u_percentage:.2f}", fontsize=fontsize)

                        plt.tight_layout()
                        
                        # Save the plot
                        plt.savefig(f'{undefined_folder}/{domain}_{split}_{count}.pdf', bbox_inches='tight')

                        # CLose
                        plt.close('all')
                        
                    if count == max_count:
                        break

    # Create a table with the structure of the experiments
    experiments = {
        "structure": pd.DataFrame(columns=files["Colored"].keys(), index=files.keys()),
        "undefined": pd.DataFrame(columns=files["Colored"].keys(), index=files.keys())
    }

    for domain, splits in domain_scores.items():
        for split, data in splits.items():
            for key, values in data.items():
                if key == "total_len": continue
                per_div = len(values)/data["total_len"]
                avg_div = np.mean([v["div"] for v in values]) if len(values) > 0 else 0
                experiments[key].at[domain, split] = f"{per_div*100:0.1f}" # ({avg_div*100:0.1f})

    # Concatenate the tables
    experiments = pd.concat(experiments, axis=1)

    # Save the table
    experiments.to_latex('paper/plots/AC/table.tex', escape=False)