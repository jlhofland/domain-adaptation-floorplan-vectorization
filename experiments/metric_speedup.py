import torch
import numpy as np
import time
import torchmetrics as tm

class CustomMetric(tm.Metric):
    def __init__(self, n_classes):
        super().__init__()
        self.add_state("confusion_matrix", default=torch.zeros((n_classes, n_classes)), dist_reduce_fx="sum")
        self.n_classes = n_classes

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

        # Increment the confusion matrix
        self.confusion_matrix += hist.reshape(self.n_classes, self.n_classes)

    def compute(self, reset=False, exclude_classes=[]):
        # Retrieve the confusion matrix tensor
        hist_tensor = self.confusion_matrix

        # Exclude classes from the confusion matrix
        # hist_tensor = hist_tensor[~torch.tensor(exclude_classes)][:, ~torch.tensor(exclude_classes)]

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

class CustomMetricNumpy:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )


    def compute(self, reset=False, exclude_classes=[]):
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        class_list = [str(i) for i in range(self.n_classes)]
        cls_acc = dict(zip(class_list, acc_cls))
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(class_list, iu))

        if reset:
            self.confusion_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

        return (
            {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
            },
            {
                "Class IoU": cls_iu,
                "Class Acc": cls_acc
            }
        )

# Parameters
batch_size = 50
img_size = 1024
n_classes = 10

# Create synthetic data
preds = torch.randint(0, n_classes, (batch_size, img_size, img_size), dtype=torch.long)
target = torch.randint(0, n_classes, (batch_size, img_size, img_size), dtype=torch.long)

# PyTorch metric computation
torch_metric = CustomMetric(n_classes=n_classes).cuda()

start_time = time.time()
for i in range(batch_size):
    torch_metric.update(preds[i].cuda(), target[i].cuda())
torch_result = torch_metric.compute()
torch_duration = time.time() - start_time

print(f"PyTorch (CUDA) Duration: {torch_duration:.6f} seconds")
print(f"PyTorch (CUDA) Result: {torch_result}")

# PyTorch metric computation
torch_metric = CustomMetric(n_classes=n_classes)

start_time = time.time()
for i in range(batch_size):
    torch_metric.update(preds[i], target[i])
torch_result = torch_metric.compute()
torch_duration = time.time() - start_time

print(f"PyTorch (CPU) Duration: {torch_duration:.6f} seconds")
print(f"PyTorch (CPU) Result: {torch_result}")

# Convert to NumPy
preds_np = preds.numpy()
target_np = target.numpy()

# NumPy metric computation
numpy_metric = CustomMetricNumpy(n_classes=n_classes)

start_time = time.time()
for i in range(batch_size):
    numpy_metric.update(preds_np[i], target_np[i])
numpy_result = numpy_metric.compute()
numpy_duration = time.time() - start_time

print(f"NumPy Duration: {numpy_duration:.6f} seconds")
print(f"NumPy Result: {numpy_result}")

# Check if they are all the same by moving to numpy and comparing
torch_result = [result.cpu().numpy() if isinstance(result, torch.Tensor) else result for result in torch_result]
