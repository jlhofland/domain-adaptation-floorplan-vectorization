import torch
from torch.nn import Module, Parameter

############################
## ADJUSTED FROM MMD LOSS ##
############################

class RBF(Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels).cuda() - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(Module):

    def __init__(self, kernel=RBF(), adaptive=False):
        super().__init__()
        self.kernel = kernel
        self.adaptive = adaptive
    
        # Certainty parameter for adaptive MMD
        self.certainty = Parameter(torch.tensor(0, requires_grad=True, dtype=torch.float32).cuda()) if adaptive else 0

        # Loss variables
        self.loss_mmd = None
        self.loss_var = None
        self.lambda_mmd = None

    def forward(self, X, Y, lambda_mmd):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()

        # Calculate the MMD loss and update the loss variable
        self.loss_mmd   = XX - 2 * XY + YY
        self.loss_var   = lambda_mmd * (self.get_adaptive_loss() if self.adaptive else self.loss_mmd)
        self.lambda_mmd = lambda_mmd * (torch.exp(-self.certainty) if self.adaptive else 1)

        # Return the MMD loss
        return self.loss_var
    
    def get_adaptive_loss(self):
        return (torch.exp(-self.certainty) * self.loss_mmd + torch.log(1 + torch.exp(self.certainty)))
    
    def get_loss(self):
        return torch.tensor([
            self.loss_mmd,
            self.loss_var,
            self.lambda_mmd,
        ])