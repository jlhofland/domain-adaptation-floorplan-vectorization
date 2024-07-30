import torch
import torchmetrics as tm
import torch.nn.functional as F
import numpy as np
from skimage import draw
from shapely.geometry import Polygon
from datasets.cubicasa5k.plotting import shp_mask

class CustomMetric(tm.Metric):
    def __init__(self, n_classes, exclude_classes=[]):
        super().__init__()
        self.add_state("confusion_matrix", default=torch.zeros((n_classes, n_classes)), dist_reduce_fx="sum")
        self.n_classes = n_classes
        self.exclude_classes = exclude_classes

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Ensure preds and target are in the correct shape, typically [batch_size, ...]
        # Flatten the predictions and targets across batch
        batch_size = preds.shape[0]
        preds = preds.view(batch_size, -1)
        target = target.view(batch_size, -1)

        # Apply the mask to select valid target indices across all batches
        mask = (target >= 0) & (target < self.n_classes)
        
        # We only want to keep data where mask is True
        preds, target = preds[mask], target[mask]

        # Calculate the indices for the flattened confusion matrix
        indices = self.n_classes * target.long() + preds.long()

        # Initialize a histogram tensor to count frequencies
        hist = torch.zeros((self.n_classes ** 2,), dtype=torch.long, device=target.device)

        # Increment the appropriate bins in the histogram
        hist.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.long))
        
        # Move the confusion matrix to the appropriate device
        if self.confusion_matrix.device != hist.device:
            self.confusion_matrix = self.confusion_matrix.to(hist.device)

        # Increment the confusion matrix
        self.confusion_matrix += hist.reshape(self.n_classes, self.n_classes)

    def compute(self, reset=False):
        # Retrieve the confusion matrix tensor
        hist_tensor = self.confusion_matrix

        # Calculate overall accuracy
        acc = torch.diag(hist_tensor).sum() / hist_tensor.sum()

        # Calculate class-wise accuracy
        acc_cls = torch.diag(hist_tensor) / hist_tensor.sum(dim=1)

        # Convert class-wise accuracy tensor to dictionary for easier access
        cls_acc = {str(i): acc_cls[i].detach() for i in range(len(acc_cls))}

        # Calculate mean accuracy
        acc_cls = torch.nanmean(acc_cls)

        # Calculate intersection over union (IoU) for each class
        iu = torch.diag(hist_tensor) / (hist_tensor.sum(dim=1) + hist_tensor.sum(dim=0) - torch.diag(hist_tensor))

        # Calculate mean IoU
        mean_iu = torch.nanmean(iu)
        
        # Calculate frequency-weighted accuracy (fwavacc)
        freq = hist_tensor.sum(dim=1) / hist_tensor.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        # Convert class-wise IoU tensor to dictionary for easier access
        cls_iu = {str(i): iu[i].detach() for i in range(len(iu))}

        # Reset the confusion matrix
        if reset:
            self.confusion_matrix.zero_()

        return (
            {
                "Overall Acc": acc.detach(),
                "Mean Acc": acc_cls.detach(),
                "FreqW Acc": fwavacc.detach(),
                "Mean IoU": mean_iu.detach(),
            },
            {
                "Class IoU": cls_iu,
                "Class Acc": cls_acc
            }
        )

    def reset(self):
        # Reset the confusion matrix
        self.confusion_matrix.zero_()

def polygons_to_tensor(polygons_val, types_val, room_polygons_val, room_types_val, size, split=[12, 11]):
    ten = np.zeros((sum(split), size[0], size[1]))

    for i, pol_type in enumerate(room_types_val):
        mask = shp_mask(room_polygons_val[i], np.arange(size[1]), np.arange(size[0]))
        ten[pol_type['class']][mask] = 1

    for i, pol_type in enumerate(types_val):
        # Index of the class
        index = pol_type['class']

        # shift the index by the number of room classes
        if pol_type['type'] == 'icon':
            index = pol_type['class'] + split[0]

        # Draw the polygon
        jj, ii = draw.polygon(polygons_val[i][:, 1], polygons_val[i][:, 0])

        # Draw the polygon if it does not go out of bounds of the tensor, else draw the polygon with the bounds
        ten[index][jj-1, ii-1] = 1

    return ten