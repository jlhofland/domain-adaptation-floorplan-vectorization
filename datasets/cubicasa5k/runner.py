import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets.cubicasa5k.metrics import CustomMetric, polygons_to_tensor
from torch import optim
from datasets.cubicasa5k.losses.mmd import MMDLoss
import wandb
import math
from datasets.cubicasa5k.post_prosessing import get_polygons
from datasets.cubicasa5k.loaders.augmentations import RotateNTurns
import numpy as np

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
        self.val_score_rooms = CustomMetric(cfg.model.input_slice[1], exclude_classes=cfg.test.exclude_classes.rooms)
        self.val_score_icons = CustomMetric(cfg.model.input_slice[2], exclude_classes=cfg.test.exclude_classes.icons)

        # Testing scores
        self.test_score_rooms = CustomMetric(cfg.model.input_slice[1], exclude_classes=cfg.test.exclude_classes.rooms)
        self.test_score_icons = CustomMetric(cfg.model.input_slice[2], exclude_classes=cfg.test.exclude_classes.icons)

        # Testing scores for polygons
        self.test_score_pol_rooms = CustomMetric(cfg.model.input_slice[1], exclude_classes=cfg.test.exclude_classes.rooms)
        self.test_score_pol_icons = CustomMetric(cfg.model.input_slice[2], exclude_classes=cfg.test.exclude_classes.icons)

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

    def _step_rotate(self, batch):
        # Transfer data to GPU
        x = batch['image']
        y = batch['label']

        # Get image size
        batch_size, channels, height, width = x.shape
        img_size = (height, width)

        # Rotate the image
        rot = RotateNTurns()
        rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
        class_count = sum(self.cfg.model.input_slice)

        # Create prediction tensor
        pred = torch.zeros([len(rotations), class_count, height, width], device=y.device)

        # For each rotation, rotate the image and predict
        for i, r in enumerate(rotations):
            forward, back = r
            # We rotate first the image
            rot_image = rot(x, 'tensor', forward)

            # We predict
            y_hat, y_lat = self.model(rot_image) # [batch_size, channels, height, width]

            # We rotate prediction back
            y_hat = rot(y_hat, 'tensor', back) 

            # We fix heatmaps
            y_hat = rot(y_hat, 'points', back)

            # We make sure the size is correct
            y_hat = F.interpolate(y_hat, size=img_size, mode='bilinear', align_corners=True)

            # We add the prediction to output
            pred[i] = y_hat[0] # [channels, height, width]

        # Calculate loss and prediction over all rotations
        y_rep = y.repeat(len(rotations), 1, 1, 1) # [pred_count, channels, height, width]
        loss  = self.loss_fn(pred, y_rep) 
        pred  = torch.mean(pred, 0, True) # [1, channels, height, width]

        # Interpolate the prediction to the original size and move to CPU
        pred = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=False) #.cpu()

        # Split the tensor into heatmaps, rooms and icons
        heats, rooms, icons = torch.split(pred, tuple(self.cfg.model.input_slice), 1)

        # Take softmax of the rooms and icons
        icons = F.softmax(icons, dim=1)
        rooms = F.softmax(rooms, dim=1)

        # Make copy of heats
        seg_heats = heats.clone()

        # Add background to the heats
        background = torch.full((batch_size, 1, height, width), self.cfg.test.heatmap_threshold, device=heats.device)
        seg_heats = torch.cat((background, seg_heats), dim=1)

        # Get segmentation classes
        seg_heats = torch.argmax(seg_heats, dim=1)
        seg_rooms = torch.argmax(rooms, dim=1)
        seg_icons = torch.argmax(icons, dim=1)

        # Get polygons of the prediction using (predictions, threshold, opening_ids)
        pol_pred = polygons_to_tensor(*get_polygons((heats, rooms, icons), threshold=self.cfg.test.heatmap_threshold, all_opening_types=[1, 2]), img_size)

        # Get the polygon segmentation classes
        rooms = torch.tensor(pol_pred[:self.cfg.model.input_slice[1]], device=seg_rooms.device).unsqueeze(0)
        icons = torch.tensor(pol_pred[self.cfg.model.input_slice[1]:], device=seg_icons.device).unsqueeze(0)
        
        # Get the polygon segmentation classes and converge to tensor
        pol_rooms = torch.argmax(rooms, dim=1)
        pol_icons = torch.argmax(icons, dim=1)

        # Remove heatmaps from the labels and move to seg_rooms and seg_icons device
        labels = y[:, self.cfg.model.input_slice[0]:].to(seg_rooms.device)

        # Return the labels, segmentation and polygon segmentation
        return (seg_heats, None), (seg_rooms, labels[:, 0]), (seg_icons, labels[:, 1]), (pol_rooms, labels[:, 0]), (pol_icons, labels[:, 1]), loss

    def _retrieve_batch(self, y_hat, y):
        # Assume y_hat and y have shapes:
        # y_hat = [batch_size, channels, height, width]
        # y = [batch_size, channels, height, width] but different channel arrangement

        # Split based on the configured slices
        input_slices = tuple(self.cfg.model.input_slice)
        heats_pred, rooms_pred, icons_pred = torch.split(y_hat, input_slices, dim=1) # [batch_size, channels, height, width]

        # Filter out the heatmaps with a certain confidence across all batches
        heats_pred = heats_pred * (heats_pred > self.cfg.test.heatmap_threshold)

        # Get the image size
        batch_size, channels, height, width = y_hat.shape

        # Add background to the heats
        background = torch.full((batch_size, 1, height, width), self.cfg.test.heatmap_threshold, device=heats_pred.device)
        heats_pred = torch.cat((background, heats_pred), dim=1)

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
        # Get labels and predictions from _step_rotate
        heats, rooms, icons, pol_rooms, pol_icons, loss = self._step_rotate(batch)
        
        # Update metrics
        self.test_score_rooms.update(*rooms)
        self.test_score_icons.update(*icons)

        # Update polygon metrics
        self.test_score_pol_rooms.update(*pol_rooms)
        self.test_score_pol_icons.update(*pol_icons)

        # Get losses
        losses = self.loss_fn.get_loss()

        # For all images in the batch, log sample
        for i in range(batch['image'].shape[0]):
            self._log_sample(*heats, *rooms, *icons, batch, i, batch_idx, "test/samples", losses, *pol_rooms, *pol_icons)

        # Clear memory
        torch.cuda.empty_cache()

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

        # Calculate polygon scores
        pol_rooms = self.test_score_pol_rooms.compute()
        pol_icons = self.test_score_pol_icons.compute()

        # Log loss and scores
        self._log_scores(*rooms, stage, "room", *pol_rooms)
        self._log_scores(*icons, stage, "icon", *pol_icons)

        # Reset test loss, and clear GPU memory
        torch.cuda.empty_cache()

    ##############################
    ##                          ##
    ##    LOGGING functions     ##
    ##                          ##
    ##############################
    # def _log_scores(self, score, class_score, stage, group, pol_score=None, pol_class_score=None):
    #     # Label group
    #     label_group = group + "_eval"

    #     # Log scores for each metric
    #     for metric, value in score.items():
    #         self.log(f"{stage}/{group}/{metric}", value)

    #     # Log scores for each class
    #     for metric, cls_dict in class_score.items():
    #         for cls, value in cls_dict.items():
    #             self.log(f"{stage}/{group}/{metric}/{cls} {self.labels[label_group][int(cls)]}", value)

    #     # Log polygon scores
    #     if pol_score is not None:
    #         data_score, data_class = [eval(self.cfg.test.experiment_measure)], []
    #         cols_score, cols_class = [self.cfg.test.experiment_variable], []

    #         # Ensure pol_score and pol_class_score have matching keys with score and class_score
    #         for metric, value in score.items():
    #             if metric in pol_score:
    #                 data_score.extend([value, pol_score[metric]])
    #                 cols_score.extend([f"{metric} (seg)", f"{metric} (vec)"])
    #             else:
    #                 raise KeyError(f"Polygon score for metric '{metric}' is missing.")

    #         for metric, cls_dict in class_score.items():
    #             data, cols = [eval(self.cfg.test.experiment_measure), metric], [self.cfg.test.experiment_variable, "metric"]
    #             for cls, value in cls_dict.items():
    #                 name_metric = self.labels[label_group][int(cls)]
    #                 data.extend([value, pol_class_score[metric].get(cls, None)])
    #                 cols.extend([f"{name_metric} (seg)", f"{name_metric} (vec)"])
    #             data_class.append(data)
    #             cols_class = cols

    #         # Log tables
    #         self.logger.experiment.log({
    #             f"{stage}/{group}/table/scores_classes": wandb.Table(data=[data_score], columns=cols_score),
    #             f"{stage}/{group}/table/scores": wandb.Table(data=data_class, columns=cols_class)
    #         })
    def _log_scores(self, score, class_score, stage, group, pol_score=None, pol_class_score=None):
        # Label group
        label_group = group + "_eval"

        # Log scores for each metric
        for metric, value in score.items():
            self.log(f"{stage}/{group}/{metric}", value)

        # Log scores for each class
        for metric, cls_dict in class_score.items():
            for cls, value in cls_dict.items():
                self.log(f"{stage}/{group}/{metric}/{cls} {self.labels[label_group][int(cls)]}", value)

        # Log polygon scores
        if pol_score is not None:
            data_score_seg, data_score_vec = [eval(self.cfg.test.experiment_measure)], [eval(self.cfg.test.experiment_measure)]
            cols_score_seg, cols_score_vec = [self.cfg.test.experiment_variable], [self.cfg.test.experiment_variable]

            data_class_seg, data_class_vec = [], []
            cols_class_seg, cols_class_vec = [], []

            # Ensure pol_score and pol_class_score have matching keys with score and class_score
            for metric, value in score.items():
                if metric in pol_score:
                    data_score_seg.extend([value])
                    data_score_vec.extend([pol_score[metric]])
                    cols_score_seg.extend([f"{metric}"])
                    cols_score_vec.extend([f"{metric}"])
                else:
                    raise KeyError(f"Polygon score for metric '{metric}' is missing.")

            for metric, cls_dict in class_score.items():
                data_seg, data_vec = [eval(self.cfg.test.experiment_measure), metric], [eval(self.cfg.test.experiment_measure), metric]
                cols_seg, cols_vec = [self.cfg.test.experiment_variable, "metric"], [self.cfg.test.experiment_variable, "metric"]
                for cls, value in cls_dict.items():
                    name_metric = self.labels[label_group][int(cls)]
                    data_seg.extend([value])
                    data_vec.extend([pol_class_score[metric].get(cls, None)])
                    cols_seg.extend([f"{name_metric}"])
                    cols_vec.extend([f"{name_metric}"])
                data_class_seg.append(data_seg)
                data_class_vec.append(data_vec)
                cols_class_seg = cols_seg
                cols_class_vec = cols_vec

            # Log tables
            self.logger.experiment.log({
                f"{stage}/{group}/table/classes_seg": wandb.Table(data=data_class_seg, columns=cols_class_seg),
                f"{stage}/{group}/table/classes_vec": wandb.Table(data=data_class_vec, columns=cols_class_vec),
                f"{stage}/{group}/table/scores_seg": wandb.Table(data=[data_score_seg], columns=cols_score_seg),
                f"{stage}/{group}/table/scores_vec": wandb.Table(data=[data_score_vec], columns=cols_score_vec)
            })



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

    def _log_sample(self, heats_pred, heats_label, rooms_pred, rooms_label, icon_pred, icon_label, batch, id, batch_idx, 
                    stage, losses, pol_rooms_pred=None, pol_rooms_label=None, pol_icons_pred=None, pol_icons_label=None):
    
        # Create class labels
        class_heats = {index: value for index, value in enumerate(self.labels["heat"])}
        class_rooms = {index: value for index, value in enumerate(self.labels["room"])}
        class_icons = {index: value for index, value in enumerate(self.labels["icon"])}

        mask_dict = {
                "room_predictions": {
                    "mask_data": rooms_pred[id].cpu().detach().numpy(),
                    "class_labels": class_rooms
                },
                "icon_predictions": {
                    "mask_data": icon_pred[id].cpu().detach().numpy(),
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
                    "mask_data": icon_label[id].cpu().detach().numpy(),
                    "class_labels": class_icons
                }
        }

        if pol_rooms_pred is not None:
            mask_dict["pol_room_predictions"] = {
                "mask_data": pol_rooms_pred[id].cpu().detach().numpy(),
                "class_labels": class_rooms
            }
            mask_dict["pol_icon_predictions"] = {
                "mask_data": pol_icons_pred[id].cpu().detach().numpy(),
                "class_labels": class_icons
            }

        image = wandb.Image(batch['image'][id], caption=f"L: {losses[1]:.2f},  R: {losses[3]:.2f}, I: {losses[6]:.2f}, H: {losses[9]:.2f}", masks=mask_dict)

        # Log room segmentation
        self.logger.experiment.log({f"{stage}/sample {id}-{batch_idx}": image})

    def calculate_lambda(self):
        return (2 / (1 + math.exp(-self.cfg.mmd.lambda_variable * (self.current_epoch / self.cfg.train.max_epochs))) - 1)
