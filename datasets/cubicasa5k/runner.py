import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets.cubicasa5k.metrics import CustomMetric
from torch import optim
import wandb

class Runner(pl.LightningModule):
    def __init__(self, cfg, model, loss_fn, labels, *args, **kwargs):
        # Initialize the LightningModule
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_fn'])

        # Set the variables
        self.cfg     = cfg
        self.model   = model
        self.loss_fn = loss_fn
        self.labels  = labels

        # Losses for logging
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
        }

        # Validation scores
        self.val_score_rooms = CustomMetric(cfg.model.input_slice[1])
        self.val_score_icons = CustomMetric(cfg.model.input_slice[2])

        # Testing scores
        self.test_score_rooms = CustomMetric(cfg.model.input_slice[1])
        self.test_score_icons = CustomMetric(cfg.model.input_slice[2])

    def forward(self, x, return_latent=False, return_output=True):
        # Runner needs to redirect any model.forward() calls to the actual network
        y_hat, y_latent = self.model(x, return_latent, return_output)

        # Return the prediction and latent space
        return y_hat, y_latent

    def configure_optimizers(self):
        # Concatinate model and loss function parameters
        params = [{'params': self.model.parameters(), 'lr': self.cfg.optimizer.lr},
                  {'params': self.loss_fn.parameters(), 'lr': self.cfg.optimizer.lr}]
        
        # Create optimizer
        optimizer = optim.Adam(params, eps=self.cfg.optimizer.eps, betas=self.cfg.optimizer.betas)

        # Create learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.cfg.optimizer.patience)

        # Return optimizer and scheduler that monitor the loss for every 1 step
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", 
                "monitor": "val/loss/all_var", # Monitor the total loss with variance
                "frequency": 1, 
            },
        }

    def _step(self, batch):
        # Transfer data to GPU
        x = batch['image']
        y = batch['label']

        # If we have target data, calculate the loss using MMD
        if 'target' in batch and self.cfg.model.use_mmd:
            # Forward pass to get prediction
            y_hat, y_latent = self.model(x, return_latent=True)

            # Get target image
            t = batch['target']

            # Forward pass to get prediction
            _, t_latent = self.model(t, return_latent=True, return_output=False)

            # Calculate the loss
            loss = self.loss_fn(y_hat, y, y_latent, t_latent)
        else:
            # Calculate prediction
            y_hat, _ = self.model(x)

            # Calculate the loss
            loss = self.loss_fn(y_hat, y)

        # Return loss and predictions
        return loss, y_hat, y
    
    def _retrieve(self, y_hat, y):
        # Select rooms and icons from the prediction
        # y_hat = [batch_size, [heatmaps, rooms, icons], height, width] = [1, [21, 13, 17], H, W]
        heats_pred, rooms_pred, icons_pred = torch.split(y_hat[0], tuple(self.cfg.model.input_slice), dim=0)
        
        # Take the argmax of the rooms and icons
        heats_pred_max = torch.argmax(heats_pred, dim=0)
        rooms_pred_max = torch.argmax(rooms_pred, dim=0)
        icons_pred_max = torch.argmax(icons_pred, dim=0)

        # Resize y to match y_hat
        y = F.interpolate(y, size=y_hat.shape[2:], mode='bilinear', align_corners=False)

        # Select rooms and icons from the label
        # y = [batch_size, [heatmaps, room_class, icon_class], height, width] = [1, [21,  1,  1], H, W]
        rooms_label = y[0, self.cfg.model.input_slice[0]]
        icons_label = y[0, self.cfg.model.input_slice[0]+1]

        # Take the argmax of heatmaps
        # heats_label = y[0, :self.cfg.model.input_slice[0]]
        # heats_label = torch.argmax(heats_label, dim=0)

        # Return the predictions and labels
        return (heats_pred_max, None), (rooms_pred_max, rooms_label), (icons_pred_max, icons_label)

    ##############################
    ##                          ##
    ##    STEPPING fuctions     ##
    ##                          ##
    ##############################
    def training_step(self, batch, batch_idx):
        # Forward pass
        loss, _, _ = self._step(batch)

        # Log step-level loss, then append to list for epoch-level loss
        self.losses["train"].append(self.loss_fn.get_loss())

        # Return loss for logging
        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass
        loss, y_hat, y = self._step(batch)

        # Retrieve predictions and labels for both rooms and icons
        heats, rooms, icons = self._retrieve(y_hat, y)

        # Update metrics
        self.val_score_rooms.update(*rooms)
        self.val_score_icons.update(*icons)

        # Log sample
        if batch_idx < 1:
            self._log_sample(*heats, *rooms, *icons, batch, batch_idx, "val/samples")

        # Log step-level loss, then append to list for epoch-level loss
        self.losses["val"].append(self.loss_fn.get_loss())

        # Return loss for logging
        return loss

    def test_step(self, batch, batch_idx):
        # Forward pass
        loss, y_hat, y = self._step(batch)

        # Retrieve predictions and labels
        heats, rooms, icons = self._retrieve(y_hat, y)

        # Update metrics
        self.test_score_rooms.update(*rooms)
        self.test_score_icons.update(*icons)

        # Log sample
        self._log_sample(*heats, *rooms, *icons, batch, batch_idx, "test/samples")

        # Log test loss
        self.losses["test"].append(self.loss_fn.get_loss())

        # Return loss for logging
        return loss

    ##############################
    ##                          ##
    ##     ENDING fuctions      ##
    ##                          ##
    ##############################
    def on_train_epoch_end(self):
        # Set stage to train
        stage = "train"

        # Calculate average training loss
        self._log_losses(stage)
        
        # Reset training loss, and clear GPU memory
        self.losses[stage] = []
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        # Set stage to validation
        stage = "val"

        # Compute scores
        rooms = self.val_score_rooms.compute(reset=True)
        icons = self.val_score_icons.compute(reset=True)

        # Log scores
        self._log_scores(*rooms, stage, "room")
        self._log_scores(*icons, stage, "icon")

        # Calculate average validation loss
        self._log_losses(stage)

        # Reset validation loss, and clear GPU memory
        self.losses[stage] = []
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        # Set stage to test
        stage = "test"

        # Calculate scores
        rooms = self.test_score_rooms.compute()
        icons = self.test_score_icons.compute()

        # Log scores
        self._log_scores(*rooms, stage, "room")
        self._log_scores(*icons, stage, "icon")

        # Calculate average test loss
        self._log_losses(stage)

        # Reset test loss, and clear GPU memory
        self.losses[stage] = []
        torch.cuda.empty_cache()

    ##############################
    ##                          ##
    ##    LOGGING functions     ##
    ##                          ##
    ##############################
    def _log_scores(self, score, class_core, stage, group):
        # Log scores for each metric
        for metric, value in score.items():
            self.log(f"{stage}/{group}/{metric}", value)

        # Log scores for each class
        for metric, dict in class_core.items():
            for cls, value in dict.items():
                self.log(f"{stage}/{group}/{metric}/{cls} {self.labels[group][int(cls)]}", value)

    def _log_losses(self, stage):
        # Stack losses and calculate average
        losses_s = torch.stack(self.losses[stage], dim=0)
        losses_m = torch.mean(losses_s, dim=0)

        # Log average losses for each label
        for i, label in enumerate(self.labels["loss"]):
            self.log(f"{stage}/loss/{label}", losses_m[i])

    def _log_sample(self, heats_pred, heats_label, rooms_pred, rooms_label, icons_pred, icons_label, batch, batch_idx, stage):
        # Create class labels
        class_heats = {index: value for index, value in enumerate(self.labels["heat"])}
        class_rooms = {index: value for index, value in enumerate(self.labels["room"])}
        class_icons = {index: value for index, value in enumerate(self.labels["icon"])}

        if (self.cfg.train.debug.print):
            # Print shapes
            print("heats_pred", heats_pred.shape)
            # print("heats_label", heats_label.shape)

            print("rooms_pred", rooms_pred.shape)
            print("rooms_label", rooms_label.shape)

            print("icons_pred", icons_pred.shape)
            print("icons_label", icons_label.shape)

            # print unique values
            print("heats_pred unique", torch.unique(heats_pred))
            # print("heats_label unique", torch.unique(heats_label))

            print("rooms_pred unique", torch.unique(rooms_pred))
            print("rooms_label unique", torch.unique(rooms_label))

            print("icons_pred unique", torch.unique(icons_pred))
            print("icons_label unique", torch.unique(icons_label))

        # Log room segmentation
        self.logger.experiment.log({f"{stage}/sample {batch_idx}": wandb.Image(batch['image'][0], masks={
            "heat_predictions": {
                "mask_data": heats_pred.cpu().detach().numpy(),
                "class_labels": class_heats
            },
            # "heat_label": {
            #     "mask_data": heats_label.cpu().detach().numpy(),
            #     "class_labels": class_heats
            # },
            "room_predictions": {
                "mask_data": rooms_pred.cpu().detach().numpy(),
                "class_labels": class_rooms
            },
            "room_label": {
                "mask_data": rooms_label.cpu().detach().numpy(),
                "class_labels": class_rooms
            },
            "icon_predictions": {
                "mask_data": icons_pred.cpu().detach().numpy(),
                "class_labels": class_icons
            },
            "icon_label": {
                "mask_data": icons_label.cpu().detach().numpy(),
                "class_labels": class_icons
            }
        })})
