import torch
import torchmetrics as tm
import torch.nn.functional as F

class CustomMetric(tm.Metric):
    def __init__(self, n_classes):
        super().__init__()
        # Initialize the confusion matrix with zeros
        self.add_state("confusion_matrix", default=torch.zeros((n_classes, n_classes), dtype=torch.long), dist_reduce_fx="sum")
        self.n_classes = n_classes

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Flatten the predictions and targets
        preds, target = preds.flatten(), target.flatten()

        # Apply the mask to select valid target indices
        mask = (target >= 0) & (target < self.n_classes)
        target, preds = target[mask], preds[mask]

        # Calculate the indices for the flattened matrix
        indices = self.n_classes * target.long() + preds.long()

        # Initialize the histogram with zeros
        hist = torch.zeros((self.n_classes ** 2,), dtype=torch.long, device=target.device)

        # Increment the appropriate bins in the histogram
        hist.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.long))

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