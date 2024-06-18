import torch
from torch.nn import Parameter, Module
from torch.nn.functional import mse_loss, cross_entropy, interpolate

class UncertaintyLoss(Module):
    def __init__(self, input_slice=[21, 13, 17],
                 target_slice=[21, 1, 1], sub=0,
                 use_cuda=True, mask=False):
        super(UncertaintyLoss, self).__init__()
        # Set the input and target slice
        self.input_slice = input_slice
        self.target_slice = target_slice

        # Set the loss variables
        self.loss = None
        self.loss_rooms = None
        self.loss_icons = None
        self.loss_heatmap = None

        # The rest
        self.mask = mask
        self.sub = sub
        self.use_cuda = use_cuda
        self.log_vars = Parameter(torch.tensor([0, 0], requires_grad=True, dtype=torch.float32).cuda())
        self.log_vars_mse = Parameter(torch.zeros(input_slice[0], requires_grad=True, dtype=torch.float32).cuda())

    def forward(self, input, target):
        # Split input and target (batch, channel, height, width)
        n, c, h, w = input.size()
        nt, ct, ht, wt = target.size()

        # Resize target to match input
        if h != ht or w != wt:  
            target = target.unsqueeze(1)
            target = interpolate(target, size=(ct, h, w), mode='nearest')
            target = target.squeeze(1)

        # Split input and target into their respective tasks
        pred_arr = torch.split(input, tuple(self.input_slice), 1)
        heatmap_pred, rooms_pred, icons_pred = pred_arr

        target_arr = torch.split(target, tuple(self.target_slice), 1)
        heatmap_target, rooms_target, icons_target = target_arr

        # Removing empty dimension if batch size is 1
        rooms_target = torch.squeeze(rooms_target, 1)
        icons_target = torch.squeeze(icons_target, 1)

        # Segmentation labels to correct type
        if self.use_cuda:
            rooms_target = rooms_target.type(torch.cuda.LongTensor) - self.sub
            icons_target = icons_target.type(torch.cuda.LongTensor) - self.sub
        else:
            rooms_target = rooms_target.type(torch.LongTensor) - self.sub
            icons_target = icons_target.type(torch.LongTensor) - self.sub

        # Calculate the loss with uncertainty magic
        self.loss_rooms_var = cross_entropy(input=rooms_pred*torch.exp(-self.log_vars[0]), target=rooms_target)
        self.loss_icons_var = cross_entropy(input=icons_pred*torch.exp(-self.log_vars[1]), target=icons_target)

        # Calculate the loss
        self.loss_rooms = cross_entropy(input=rooms_pred, target=rooms_target)
        self.loss_icons = cross_entropy(input=icons_pred, target=icons_target)

        # If we want to mask the heatmap loss
        # if self.mask:
        #     heatmap_mask = rooms_pred
        #     self.loss_heatmap_var, self.vars_sum, self.loss_heatmap = self.homosced_heatmap_mse_loss_mask(heatmap_pred, heatmap_target, heatmap_mask, self.log_vars_mse)
        # else:

        # If we want to mask the heatmap loss
        self.loss_heatmap_var = self.homosced_heatmap_mse_loss(heatmap_pred, heatmap_target, self.log_vars_mse)
        self.loss_heatmap = mse_loss(input=heatmap_pred, target=heatmap_target)

        # Add the losses together to get the total loss
        self.loss     = self.loss_rooms     + self.loss_icons     + self.loss_heatmap
        self.loss_var = self.loss_rooms_var + self.loss_icons_var + self.loss_heatmap_var

        return self.loss_var

    def homosced_heatmap_mse_loss(self, input, target, logvars):
        # we have n heatmaps, i.e. n heatmap tasks
        n, ntasks, h, w = input.size()

        # make a 2d tensor from both input and target  so that we have n tasks cols
        preds = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, ntasks)
        targets = target.transpose(1, 2).transpose(2, 3).contiguous().view(-1, ntasks)

        # take elementwise subtraction and raise to the power of two
        diff = (preds - targets) ** 2

        # measure task dependent mse loss
        mse_loss_per_tasks = torch.sum(diff, 0) / (n * h * w)

        # apply uncertainty magic
        # w_mse_loss = torch.exp(-logvars) * mse_loss_per_tasks + logvars
        w_mse_loss = torch.exp(-logvars) * mse_loss_per_tasks + torch.log(1+torch.exp(logvars))

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
            loss_params[2]
        ]

        # Convert the list of loss values to a tensor
        return torch.tensor(loss_values)

