import torch
from torch.nn import Parameter, Module
from torch.nn.functional import mse_loss, cross_entropy, interpolate

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

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

class UncertaintyLoss(Module):
    def __init__(self, input_slice=[21, 13, 17],
                 target_slice=[21, 1, 1], sub=0,
                 use_cuda=True, mask=False, use_mmd=False):
        super(UncertaintyLoss, self).__init__()
        # Set the input and target slice
        self.input_slice = input_slice
        self.label_slice = target_slice

        # Set the loss variables
        self.loss = None
        self.loss_rooms = None
        self.loss_icons = None
        self.loss_heatmap = None
        self.loss_mmd = 0

        # Init mmd uncertainty with 0 if not used
        self.loss_mmd_var = 0

        # Mask, sub, and cuda
        self.mask = mask
        self.sub = sub
        self.use_cuda = use_cuda
        self.use_mmd = use_mmd

        # MMD loss
        if use_mmd:
            self.mmd_loss = MMDLoss()
            # self.mmd_vars = Parameter(torch.tensor([0], requires_grad=True, dtype=torch.float32).cuda())
            self.mmd_vars = Parameter(torch.tensor([0], requires_grad=True, dtype=torch.float32).cuda())

        # Uncertainty parameters
        self.log_vars = Parameter(torch.tensor([0, 0], requires_grad=True, dtype=torch.float32).cuda())
        self.log_vars_mse = Parameter(torch.zeros(input_slice[0], requires_grad=True, dtype=torch.float32).cuda())

    def forward(self, input, label, source_latent=None, target_latent=None):
        # Split input and target (batch, channel, height, width)
        n, c, h, w = input.size()
        nl, cl, hl, wl = label.size()

        # Resize label to match input
        if h != hl or w != wl:  
            label = label.unsqueeze(1)
            label = interpolate(label, size=(cl, h, w), mode='nearest')
            label = label.squeeze(1)

        # Split input and target into their respective tasks
        heatmap_input, rooms_input, icons_input = torch.split(input, tuple(self.input_slice), 1)
        heatmap_label, rooms_label, icons_label = torch.split(label, tuple(self.label_slice), 1)

        # Removing empty dimension if batch size is 1
        rooms_label = torch.squeeze(rooms_label, 1)
        icons_label = torch.squeeze(icons_label, 1)

        # Segmentation labels to correct type
        if self.use_cuda:
            rooms_label = rooms_label.type(torch.cuda.LongTensor) - self.sub
            icons_label = icons_label.type(torch.cuda.LongTensor) - self.sub
        else:
            rooms_label = rooms_label.type(torch.LongTensor) - self.sub
            icons_label = icons_label.type(torch.LongTensor) - self.sub

        # Calculate the MMD loss if provided
        if self.use_mmd and source_latent is not None and target_latent is not None:
            # Flatten the feature maps to use in MMD
            # [batch_size, C, H/64, W/64] --> [batch_size, C*H/64*W/64]
            source_latent = source_latent.view(source_latent.size(0), -1)
            target_latent = target_latent.view(target_latent.size(0), -1)

            # Calculate the MMD loss
            self.loss_mmd     = self.mmd_loss(source_latent, target_latent)
            # self.loss_mmd_var = self.mmd_loss(source_latent*torch.exp(-self.mmd_vars), target_latent)
            self.loss_mmd_var = torch.exp(-self.mmd_vars) * self.loss_mmd + torch.log(1 + torch.exp(self.mmd_vars))
            # self.loss_mmd_var = self.mmd_vars * self.loss_mmd

        # Calculate the loss with uncertainty magic
        self.loss_rooms_var = cross_entropy(input=rooms_input*torch.exp(-self.log_vars[0]), target=rooms_label)
        self.loss_icons_var = cross_entropy(input=icons_input*torch.exp(-self.log_vars[1]), target=icons_label)

        # Calculate the loss
        self.loss_rooms = cross_entropy(input=rooms_input, target=rooms_label)
        self.loss_icons = cross_entropy(input=icons_input, target=icons_label)

        # If we want to mask the heatmap loss
        # if self.mask:
        #     heatmap_mask = rooms_pred
        #     self.loss_heatmap_var, self.vars_sum, self.loss_heatmap = self.homosced_heatmap_mse_loss_mask(heatmap_pred, heatmap_target, heatmap_mask, self.log_vars_mse)
        # else:

        # If we want to mask the heatmap loss
        self.loss_heatmap_var = self.homosced_heatmap_mse_loss(heatmap_input, heatmap_label, self.log_vars_mse)
        self.loss_heatmap = mse_loss(input=heatmap_input, target=heatmap_label)

        # Add the losses together to get the total loss
        self.loss     = self.loss_rooms     + self.loss_icons     + self.loss_heatmap     + self.loss_mmd
        self.loss_var = self.loss_rooms_var + self.loss_icons_var + self.loss_heatmap_var + self.loss_mmd_var

        return self.loss_var

    def homosced_heatmap_mse_loss(self, input, target, logvars):
        # we have n heatmaps, i.e. n heatmap tasks
        n, ntasks, h, w = input.size()

        # make a 2d tensor from both input and target so that we have n tasks cols
        preds = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, ntasks)
        targets = target.transpose(1, 2).transpose(2, 3).contiguous().view(-1, ntasks)

        # take elementwise subtraction and raise to the power of two
        diff = (preds - targets) ** 2

        # measure task dependent mse loss
        mse_loss_per_tasks = torch.sum(diff, 0) / (n * h * w)

        # apply uncertainty magic
        w_mse_loss = torch.exp(-logvars) * mse_loss_per_tasks + torch.log(1 + torch.exp(logvars))

        # take sum and return it
        w_mse_loss_total = w_mse_loss.sum()

        return w_mse_loss_total

    def get_loss(self):
        # List of loss parameters
        loss_params = [
            torch.exp(-self.log_vars[0]), # rooms
            torch.exp(-self.log_vars[1]), # icons
            torch.exp(-self.log_vars_mse) # heatmaps
        ]

        # Sum weights of heatmaps
        loss_params[2] = loss_params[2].sum()

        # # Include MMD loss parameter in the list of loss parameters
        # if self.use_mmd:
        #     loss_params.append(self.log_vars[2]) # MMD uncertainty parameter

        # List of loss variables [5, 5, 3]
        loss_values = [
            # total loss
            self.loss,
            self.loss_var,
            # rooms
            self.loss_rooms,
            self.loss_rooms_var,
            loss_params[0],
            # icons
            self.loss_icons,
            self.loss_icons_var,
            loss_params[1],
            # heatmap
            self.loss_heatmap,
            self.loss_heatmap_var,
            loss_params[2],
            # mmd
            self.loss_mmd,
            self.loss_mmd_var,
            torch.exp(-self.mmd_vars) if self.use_mmd else 0
        ]

        # Convert the list of loss values to a tensor
        return torch.tensor(loss_values)

