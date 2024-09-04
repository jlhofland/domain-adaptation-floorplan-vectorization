import torch
from omegaconf import OmegaConf
from datasets.cubicasa5k.loaders.svg_loader import FloorplanSVG
from model_factory import factory
import matplotlib.pyplot as plt
import tqdm
from datasets.cubicasa5k.loaders.augmentations import DictToTensor, Compose
import numpy as np

from datasets.cubicasa5k.metrics import CustomMetric, polygons_to_tensor
from datasets.cubicasa5k.post_prosessing import get_polygons

import torch.nn.functional as F
from datasets.cubicasa5k.loaders.augmentations import RotateNTurns


colors = {
    "Rooms": ['#DCDCDC', '#b3de69', '#262626', '#8dd3c7', '#fdb462', '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969', '#577a4d', '#ffffb3'],
    "Icons": ['#DCDCDC', '#8dd3c7', '#b15928', '#fdb462', '#ffff99', '#fccde5', '#80b1d3', '#808080', '#fb8072', '#696969', '#577a4d']
}

files = {
    "A": "test_hqa.txt",
    "H": "test_hq.txt",
    "C": "test_c.txt"
}
labels = {
    "Rooms": ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"],
    "Icons": ["No Icon", "Window", "Door", "Closet", "Elect. Appl." ,"Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]
}
fontsize = 18
font_sm  = 12
model_weights = "weights/model_best_val_loss_var.pkl"

# Load defaults and overwrite by command-line arguments
cfg = OmegaConf.load("config_pkl.yaml")
cmd_cfg = OmegaConf.from_cli()
cfg = OmegaConf.merge(cfg, cmd_cfg)
print(OmegaConf.to_yaml(cfg))

domain_scores = {}
class_counts  = {}

for i, (domain, domain_txt) in enumerate(files.items()):
    # Load data
    dataset = FloorplanSVG(domain_txt, cfg, pre_load=True, augmentations=Compose([DictToTensor()]))
    loader  = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    # Load model weights
    weights = torch.load(model_weights, map_location=torch.device('cpu'))

    # Create model
    model = factory(cfg)

    # Load model weights
    model.load_state_dict(weights['model_state'])

    # Set model to evaluation mode
    model.eval()

    # Testing scores
    score_rooms = CustomMetric(cfg.model.input_slice[1])
    score_icons = CustomMetric(cfg.model.input_slice[2])

    # Testing scores for polygons
    score_pol_rooms = CustomMetric(cfg.model.input_slice[1])
    score_pol_icons = CustomMetric(cfg.model.input_slice[2])

    class_counts[domain] = {
        "Rooms": {class_name: 0 for class_name in labels["Rooms"]},
        "Icons": {class_name: 0 for class_name in labels["Icons"]}
    }

    # Loop over data
    with torch.no_grad():
        for count, val in tqdm.tqdm(enumerate(loader), total=len(loader), desc=f"Domain: {domain}", ncols=80, leave=False):
            # Unpack data
            image = val['image']
            label = val['label']

            # Forward pass
            output = model(image)

            # Img size
            batch, _, height, width = image.shape

            # Create rotations
            rot_class = RotateNTurns()
            rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]

            # Create prediction tensor
            pred_count = len(rotations)
            prediction = torch.zeros([pred_count, sum(cfg.model.input_slice), height, width])

            # Loop over rotations
            for i, r in enumerate(rotations):
                forward, back = r
                # We rotate first the image
                rot_image = rot_class(image, 'tensor', forward)
                pred, latent = model(rot_image)

                # We rotate prediction back
                pred = rot_class(pred, 'tensor', back)

                # We fix heatmaps
                pred = rot_class(pred, 'points', back)

                # We make sure the size is correct
                pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)

                # We add the prediction to output
                prediction[i] = pred[0]

            # We average the predictions
            prediction = torch.mean(prediction, dim=0, keepdim=True)

            # Interpolate the prediction to the original size and move to CPU
            prediction = F.interpolate(prediction, size=(height, width), mode='bilinear', align_corners=False) #.cpu()

            # Split the tensor into heatmaps, rooms and icons
            heats, rooms, icons = torch.split(prediction, tuple(cfg.model.input_slice), 1)

            # Take softmax of the rooms and icons
            icons = F.softmax(icons, dim=1)
            rooms = F.softmax(rooms, dim=1)

            # Get segmentation classes
            seg_rooms = torch.argmax(rooms, dim=1)
            seg_icons = torch.argmax(icons, dim=1)

            # Get polygons of the prediction using (predictions, threshold, opening_ids)
            pol_pred = polygons_to_tensor(*get_polygons((heats, rooms, icons), threshold=cfg.test.heatmap_threshold, all_opening_types=[1, 2]), (height, width))

            # Get the polygon segmentation classes
            rooms = torch.tensor(pol_pred[:cfg.model.input_slice[1]]).unsqueeze(0)
            icons = torch.tensor(pol_pred[cfg.model.input_slice[1]:]).unsqueeze(0)
            
            # Get the polygon segmentation classes and converge to tensor
            pol_rooms = torch.argmax(rooms, dim=1)
            pol_icons = torch.argmax(icons, dim=1)

            # Get room and icon labels
            label = label[:, cfg.model.input_slice[0]:]

            # Update scores
            score_rooms.update(seg_rooms, label[:, 0])
            score_icons.update(seg_icons, label[:, 1])
            score_pol_rooms.update(pol_rooms, label[:, 0])
            score_pol_icons.update(pol_icons, label[:, 1])

            # Add 1 to the class counts if the label contains at least one pixel
            class_counts[domain]["Rooms"] = {class_name: class_counts[domain]["Rooms"][class_name] + 1 if torch.sum(label[:, 0] == i).item() > 0 else class_counts[domain]["Rooms"][class_name] for i, class_name in enumerate(labels["Rooms"])}
            class_counts[domain]["Icons"] = {class_name: class_counts[domain]["Icons"][class_name] + 1 if torch.sum(label[:, 1] == i).item() > 0 else class_counts[domain]["Icons"][class_name] for i, class_name in enumerate(labels["Icons"])}

    # Compute the scores
    general_rooms, classes_rooms = score_rooms.compute()
    general_icons, classes_icons = score_icons.compute()

    # Compute the polygon scores
    general_room_vec, classes_room_vec = score_pol_rooms.compute()
    general_icon_vec, classes_icon_vec = score_pol_icons.compute()

    # Save the scores
    domain_scores[domain] = {
        "Rooms": {
            "Segmentation": {
                "general_dict": general_rooms,
                "classes_dict": classes_rooms
            },
            "Vectorization": {
                "general_dict": general_room_vec,
                "classes_dict": classes_room_vec
            }
        },
        "Icons": {
            "Segmentation": {
                "general_dict": general_icons,
                "classes_dict": classes_icons,
            },
            "Vectorization": {
                "general_dict": general_icon_vec,
                "classes_dict": classes_icon_vec,
            }
        }
    }

# Create plots
fig_rooms, ax_rooms = plt.subplots(1, len(labels["Rooms"])+1, figsize=(24, 3), sharey=True)
fig_icons, ax_icons = plt.subplots(1, len(labels["Icons"])+1, figsize=(24, 3), sharey=True)

# Create plots
plots = {
    "Rooms": {
        "fig": fig_rooms,
        "ax": ax_rooms
    },
    "Icons": {
        "fig": fig_icons,
        "ax": ax_icons,
    }
}

# Multiply all class scores
for domain, domain_data in domain_scores.items():
    for class_type, class_data in domain_data.items():
        for data_type, data_file in class_data.items():
            for class_name, class_scores in data_file["classes_dict"]["Class IoU"].items():
                data_file["classes_dict"]["Class IoU"][class_name] *= 100
            data_file["general_dict"]["Mean IoU"] *= 100

# Loop over the domain scores
for i, (domain, domain_data) in enumerate(domain_scores.items()): # High Quality Architectural, High Quality, Colored
    for j, (class_type, class_data) in enumerate(domain_data.items()): # Rooms, Icons
        for k, (data_type, data_file) in enumerate(class_data.items()): # Segmentation, Vectorization
            for l, (class_name, class_scores) in enumerate(data_file["classes_dict"]["Class IoU"].items()):
                # Get the class scores
                segmentation_scores = class_data["Segmentation"]["classes_dict"]["Class IoU"][class_name]
                vectorization_scores = class_data["Vectorization"]["classes_dict"]["Class IoU"][class_name]

                # Check if Segmentation or Vectorization is higher for class l
                if segmentation_scores > vectorization_scores:
                    text_aligning, text_pad = ["center", "center"], [6, -6]
                else:
                    text_aligning, text_pad = ["center", "center"], [-6, 6]

                plots[class_type]["ax"][l].bar(domain, class_scores, color=colors[class_type][l], alpha=0.5, edgecolor='black', linestyle='dashed' if data_type == "Vectorization" else None)

                # If vectorization, set text below the top of the bar with the iou score
                if data_type == "Vectorization" and class_scores > 0:
                    plots[class_type]["ax"][l].text(domain, class_scores - text_pad[0], f"{class_scores:.1f}", ha='center', va=text_aligning[0], fontsize=font_sm)
                elif class_scores > 0:
                    # If segmentation, set text above the top of the bar with the iou score
                    plots[class_type]["ax"][l].text(domain, class_scores - text_pad[1], f"{class_scores:.1f}", ha='center', va=text_aligning[1], fontsize=font_sm)
                
                # Set x-axis to fontsize
                plots[class_type]["ax"][l].tick_params(axis='both', which='major', labelsize=fontsize)

                # Add the class name to the plot
                plots[class_type]["ax"][l].set_title(labels[class_type][l], fontsize=fontsize, pad=20)

                # Set Y-axis to 0-1
                plots[class_type]["ax"][l].set_ylim(0, 110)

                # Set y-ticks to 0.2
                plots[class_type]["ax"][l].set_yticks(np.arange(0, 110, 20))

                # Add a border around the each bar in the plot
                for spine in plots[class_type]["ax"][l].spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1)

            # Check if Segmentation or Vectorization is higher for class l
            if class_data["Segmentation"]["general_dict"]["Mean IoU"] > class_data["Vectorization"]["general_dict"]["Mean IoU"]:
                text_aligning, text_pad = ["center", "center"], [6, -6]
            else:
                text_aligning, text_pad = ["center", "center"], [-6, 6]

            # If vectorization, set text below the top of the bar with the iou score
            if data_type == "Vectorization":
                # Plot the general IoU
                plots[class_type]["ax"][-1].bar(domain, data_file["general_dict"]["Mean IoU"], color="black", alpha=0.5, edgecolor='black', linestyle='dashed')
                plots[class_type]["ax"][-1].text(domain, data_file["general_dict"]["Mean IoU"] - text_pad[0], f"{data_file['general_dict']['Mean IoU']:.1f}", ha='center', va=text_aligning[0], fontsize=font_sm)
            else:
                # If segmentation, set text above the top of the bar with the iou score
                plots[class_type]["ax"][-1].bar(domain, data_file["general_dict"]["Mean IoU"], color="black", alpha=0.5, edgecolor='black')
                plots[class_type]["ax"][-1].text(domain, data_file["general_dict"]["Mean IoU"] - text_pad[1], f"{data_file['general_dict']['Mean IoU']:.1f}", ha='center', va=text_aligning[1], fontsize=font_sm)

            # Set x-axis to fontsize
            plots[class_type]["ax"][-1].tick_params(axis='both', which='major', labelsize=fontsize)

            # Add the class name to the
            plots[class_type]["ax"][-1].set_title(class_type, fontsize=fontsize, pad=20)

# Add the class counts per domain above the subplots
for i, (domain, domain_data) in enumerate(class_counts.items()):
    for j, (class_type, class_data) in enumerate(domain_data.items()):
        class_data[class_type] = sum(class_data.values())
        for k, (class_name, class_count) in enumerate(class_data.items()):
            plots[class_type]["ax"][k].text(i, plots[class_type]["ax"][k].get_ybound()[1] + 0.01, f'{class_count}', ha='center', va='bottom', fontsize=font_sm)

# Set y-labels
ax_rooms[0].set_ylabel("IoU (%)", fontsize=fontsize)
ax_icons[0].set_ylabel("IoU (%)", fontsize=fontsize)

# Rotate the x-axis labels
for plot in plots.values():
    for ax in plot["ax"]:
        ax.set_xticklabels(files.keys())

# Tight layout
fig_rooms.tight_layout()
fig_icons.tight_layout()

# Save the plots
fig_rooms.savefig("paper/plots/domain_performance_rooms.pdf", bbox_inches='tight')
fig_icons.savefig("paper/plots/domain_performance_icons.pdf", bbox_inches='tight')