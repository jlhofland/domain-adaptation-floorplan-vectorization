import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

class Runner(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        # Runner needs to redirect any model.forward() calls to the actual network
        return self.model(x)

    def configure_optimizers(self):
        if self.cfg.optimize.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimize.lr)
        else:
            raise NotImplementedError(f"Optimizer {self.cfg.optimizer}")
        return optimizer

    def _step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, y_hat = self._step(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.train_accuracy(preds, batch[1])

        # Log step-level loss & accuracy
        self.log("train/loss_step", loss)
        self.log("train/acc_step", self.train_accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self._step(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.val_accuracy(preds, batch[1])

        # Log step-level loss & accuracy
        self.log("val/loss_step", loss)
        self.log("val/acc_step", self.val_accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat = self._step(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.test_accuracy(preds, batch[1])

        # Log test loss
        self.log("test/loss", loss)
        self.log('test/acc', self.test_accuracy)
        return loss

    def on_train_epoch_end(self):
        # Log the epoch-level training accuracy
        self.log('train/acc', self.train_accuracy.compute())
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        # Log the epoch-level validation accuracy
        self.log('val/acc', self.val_accuracy.compute())
        self.val_accuracy.reset()