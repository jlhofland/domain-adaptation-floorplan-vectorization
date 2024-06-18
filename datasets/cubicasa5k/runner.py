import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets.cubicasa5k.metrics import CustomMetric
from torch import optim
from datasets.cubicasa5k.losses.mmd import MMDLoss
import wandb
import math

class Runner(pl.LightningModule):
    def __init__(self, cfg, model, loss_fn, labels, *args, **kwargs):
        # Initialize the LightningModule
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_fn', 'loss_mmd'])

        # Set the variables
        self.cfg      = cfg
        self.model    = model
        self.loss_fn  = loss_fn
        self.loss_mmd = MMDLoss(adaptive=cfg.mmd.lambda_adaptive)
        self.labels   = labels
        self.mmd_lambda = self.cfg.mmd.lambda_constant * (self.calculate_lambda() if self.cfg.mmd.lambda_variable else 1)

        # Create a 2 lists for source and target latent data, result should be [batch_size, C, H, W]
        self.source_latents = []
        self.target_latents = []

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
        
        # If we use MMD, also add the MMD loss parameters
        if self.cfg.mmd.enable and self.cfg.mmd.lambda_adaptive:
            params.append({'params': self.loss_mmd.parameters(), 'lr': self.cfg.optimizer.lr})
        
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
                "monitor": "val/loss/all/uncertainty", # Monitor the total loss with variance
                "frequency": 1,
            },

        }

    def _step(self, batch):
        # Transfer data to GPU
        x = batch['image']
        y = batch['label']

        # Calculate the prediction and latent space
        y_hat, y_lat = self.model(x, return_latent=1)

        # Calculate the loss
        loss = self.loss_fn(y_hat, y)
        
        # If we have target data, also get the latent space representations
        if 'target' in batch and self.cfg.mmd.enable:
            _, t_lat = self.model(batch['target'], return_latent=1, return_output=0)

            # Flatten: [batch_size, C, H/64, W/64] --> [batch_size, C*H/64*W/64]
            y_lat = y_lat.view(y_lat.size(0), -1)
            t_lat = t_lat.view(t_lat.size(0), -1)
            
            # If we have more than one sample in the batch, calculate the MMD loss (required for MMD loss)
            if x.shape[0] > 1:
                loss += self.loss_mmd(y_lat, t_lat, self.mmd_lambda)
            else:
                self.source_latents.append(y_lat)
                self.target_latents.append(t_lat)

        # Return loss and predictions
        return loss, y_hat, y
    
    def _retrieve(self, y_hat, y):
        # Select rooms and icons from the prediction
        # y_hat = [batch_size, [heatmaps, rooms, icons], height, width] = [1, [21, 13, 17], H, W]
        heats_pred, rooms_pred, icons_pred = torch.split(y_hat[0], tuple(self.cfg.model.input_slice), dim=0)

        # Filter out the heatmaps with > 0.8 confidence
        heats_pred = heats_pred * (heats_pred > 0.5)
        
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

        # Return the predictions and labels
        return (heats_pred_max, None), (rooms_pred_max, rooms_label), (icons_pred_max, icons_label)

    def _retrieve_batch(self, y_hat, y):
        # Assume y_hat and y have shapes:
        # y_hat = [batch_size, channels, height, width]
        # y = [batch_size, channels, height, width] but different channel arrangement

        # Split based on the configured slices
        input_slices = tuple(self.cfg.model.input_slice)
        heats_pred, rooms_pred, icons_pred = torch.split(y_hat, input_slices, dim=1)

        # Filter out the heatmaps with > 0.5 confidence across all batches
        heats_pred = heats_pred * (heats_pred > 0.5)

        # Take the argmax of the rooms and icons across channel dimensions
        heats_pred_max = torch.argmax(heats_pred, dim=1)
        rooms_pred_max = torch.argmax(rooms_pred, dim=1)
        icons_pred_max = torch.argmax(icons_pred, dim=1)

        # Resize y to match y_hat dimensions, applying interpolation across the batch
        y_resized = F.interpolate(y, size=y_hat.shape[2:], mode='bilinear', align_corners=False)

        # Extract rooms and icons labels using the specified slices
        rooms_label = y_resized[:, self.cfg.model.input_slice[0], :, :]
        icons_label = y_resized[:, self.cfg.model.input_slice[0]+1, :, :]

        # Return predictions and labels for heats, rooms, and icons
        # Here we return tuples for each category where:
        # - The first element of the tuple is the prediction map
        # - The second element of the tuple is the ground truth label map
        return (heats_pred_max, None), (rooms_pred_max, rooms_label), (icons_pred_max, icons_label)

    ##############################
    ##                          ##
    ##    STEPPING fuctions     ##
    ##                          ##
    ##############################
    def on_train_epoch_start(self):
        # Update MMD lamdba if we use variable lambda
        self.mmd_lambda = self.cfg.mmd.lambda_constant * (self.calculate_lambda() if self.cfg.mmd.lambda_variable else 1)

    def training_step(self, batch, batch_idx):
        # Forward pass
        loss, _, _ = self._step(batch)

        # Get losses
        losses = self.loss_fn.get_loss()

        # If we use MMD add the MMD loss to the total loss
        if self.cfg.mmd.enable:
            # Get the MMD losses
            mmd_losses = self.loss_mmd.get_loss()

            # Append the MMD losses to the losses list
            losses = torch.cat((losses, mmd_losses))

            # Add loss and scaled loss to the total loss
            losses[0] += mmd_losses[0]
            losses[1] += mmd_losses[1]

        # Log step-level loss, then append to list for epoch-level loss
        self.losses["train"].append(losses)

        # Return loss for logging
        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass
        loss, y_hat, y = self._step(batch)

        # Retrieve predictions and labels for both rooms and icons
        heats, rooms, icons = self._retrieve_batch(y_hat, y)

        # Update metrics
        self.val_score_rooms.update(*rooms)
        self.val_score_icons.update(*icons)

        # Get losses
        losses = self.loss_fn.get_loss()

        # For lenth of batch, log sample
        if batch_idx < 3:
            self._log_sample(*heats, *rooms, *icons, batch, 0, batch_idx, "val/samples", losses)

        # Get losses and weights
        self.losses["val"].append(losses)

        # Return loss for logging
        return loss

    def test_step(self, batch, batch_idx):
        # Forward pass
        loss, y_hat, y = self._step(batch)

        # Retrieve predictions and labels
        heats, rooms, icons = self._retrieve_batch(y_hat, y)

        # Update metrics
        self.test_score_rooms.update(*rooms)
        self.test_score_icons.update(*icons)

        # Get losses
        losses = self.loss_fn.get_loss()

        # For all images in the batch, log sample
        for i in range(batch['image'].shape[0]):
            self._log_sample(*heats, *rooms, *icons, batch, i, batch_idx, "test/samples", losses)

        # Get losses and weights
        self.losses["test"].append(losses)

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
        
        # Log loss
        self._log_loss(stage)
        
        # Reset training loss, and clear GPU memory
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        # Set stage to validation
        stage = "val"

        # Compute scores
        rooms = self.val_score_rooms.compute(reset=True)
        icons = self.val_score_icons.compute(reset=True)

        # Log scores and loss
        self._log_loss(stage)
        self._log_scores(*rooms, stage, "room")
        self._log_scores(*icons, stage, "icon")

        # Reset validation loss, and clear GPU memory
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        # Set stage to test
        stage = "test"

        # Calculate scores
        rooms = self.test_score_rooms.compute()
        icons = self.test_score_icons.compute()

        # Log loss and scores
        self._log_loss(stage)
        self._log_scores(*rooms, stage, "room")
        self._log_scores(*icons, stage, "icon")

        # Reset test loss, and clear GPU memory
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

    def _log_loss(self, stage):
        # Stack losses and calculate average
        ccd_losses = torch.stack(self.losses[stage], dim=0)
        avg_losses = torch.mean(ccd_losses, dim=0)

        # If we use MMD, calculate the MMD loss and add it to the total loss
        if stage in ("val") and self.cfg.mmd.enable:
            # Stack source and target latent data
            mmd_sources = torch.cat(self.source_latents, dim=0)
            mmd_targets = torch.cat(self.target_latents, dim=0)

            # Calculate the MMD loss over entire epoch
            self.loss_mmd(mmd_sources, mmd_targets, self.mmd_lambda)

            # Get losses and concatenate to averages
            mmd_losses = self.loss_mmd.get_loss()
            avg_losses = torch.cat((avg_losses, mmd_losses))

            # Add mmd parts to the total loss
            avg_losses[0] += mmd_losses[0]
            avg_losses[1] += mmd_losses[1]

        # Zip labels["loss"] and average to log
        self.log_dict({stage + "/" + metric: value for metric, value in zip(self.labels["loss"], avg_losses)})

        # Reset losses
        self.losses[stage]  = []
        self.source_latents = []
        self.target_latents = []

    def _log_sample(self, heats_pred, heats_label, rooms_pred, rooms_label, icons_pred, icons_label, batch, id, idx, stage, ls):
        # Create class labels
        class_heats = {index: value for index, value in enumerate(self.labels["heat"])}
        class_rooms = {index: value for index, value in enumerate(self.labels["room"])}
        class_icons = {index: value for index, value in enumerate(self.labels["icon"])}

        image = wandb.Image(
            batch['image'][id], 
            caption=f"L: {ls[1]:.2f},  R: {ls[3]:.2f}, I: {ls[6]:.2f}, H: {ls[9]:.2f}",
            masks={
                "room_predictions": {
                    "mask_data": rooms_pred[id].cpu().detach().numpy(),
                    "class_labels": class_rooms
                },
                "icon_predictions": {
                    "mask_data": icons_pred[id].cpu().detach().numpy(),
                    "class_labels": class_icons
                },
                "heat_predictions": {
                    "mask_data": heats_pred[id].cpu().detach().numpy(),
                    "class_labels": class_heats
                },
                "room_label": {
                    "mask_data": rooms_label[id].cpu().detach().numpy(),
                    "class_labels": class_rooms
                },
                "icon_label": {
                    "mask_data": icons_label[id].cpu().detach().numpy(),
                    "class_labels": class_icons
                }
        })

        # Log room segmentation
        self.logger.experiment.log({f"{stage}/sample {id}-{idx}": image})

    def calculate_lambda(self):
        return (2 / (1 + math.exp(-self.cfg.mmd.lambda_variable * (self.current_epoch / self.cfg.train.max_epochs))) - 1)
