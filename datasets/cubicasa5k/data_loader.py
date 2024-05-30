import torch
import pytorch_lightning as pl

# Import cubicasa5k dataset
from datasets.cubicasa5k.loaders.svg_loader_mmd import FloorplanSVGMMD
from datasets.cubicasa5k.loaders.svg_loader import FloorplanSVG
from datasets.cubicasa5k.loaders.augmentations import (RandomCropToSizeTorch, ResizePaddedTorch, Compose, DictToTensor, ColorJitterTorch, RandomRotations)
from datasets.cubicasa5k.loaders.augmentations_mmd import (RandomCropToSizeTorchMMD, ResizePaddedTorchMMD, DictToTensorMMD, RandomRotationsMMD)
from torchvision.transforms import RandomChoice
from torch.nn import functional as F


class CubiCasa5K(pl.LightningDataModule):
    def __init__(self, cfg):
        # Call super constructor
        super().__init__()

        # Define the configuration
        self.cfg = cfg
        
        if cfg.model.use_mmd:
            self.svg_loader = FloorplanSVGMMD
            self.random_crop = RandomCropToSizeTorchMMD
            self.resize_padded = ResizePaddedTorchMMD
            self.random_rotations = RandomRotationsMMD
            self.dict_tensor = DictToTensorMMD
        else:
            self.svg_loader = FloorplanSVG
            self.random_crop = RandomCropToSizeTorch
            self.resize_padded = ResizePaddedTorch
            self.random_rotations = RandomRotations
            self.dict_tensor = DictToTensor

        # Get image size
        size = (cfg.dataset.image_size, cfg.dataset.image_size)
        vals = (cfg.dataset.val_size, cfg.dataset.val_size)

        # Set scaling and cropping augmentations
        scale_augmentations = RandomChoice([
            self.random_crop(data_format='dict', size=size),
            self.resize_padded((0, 0), data_format='dict', size=size)
        ]) if cfg.dataset.scale else self.random_crop(data_format='dict', size=size)

        # Set training augmentations to apply
        self.augmentations = Compose([
            scale_augmentations,
            self.random_rotations(format='cubi'),
            self.dict_tensor(),
            ColorJitterTorch(gray=cfg.dataset.grayscale)
        ]) if cfg.dataset.augmentations else self.dict_tensor()

        self.val_augmentations = Compose([
            self.resize_padded((0, 0), data_format='dict', size=vals),
            self.dict_tensor()
        ]) if cfg.dataset.val_size else self.dict_tensor()


    def train_data(self):
        return torch.utils.data.DataLoader(
            self.svg_loader(
                augmentations=self.augmentations,
                cfg=self.cfg,
                pre_load=True,
                source_list=self.cfg.dataset.files.train,
                target_list=self.cfg.dataset.files.mmd.train,
            ),
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=self.cfg.dataset.pin_memory,
            persistent_workers=self.cfg.dataset.persistent_workers,
        )

    def train_dataloader(self):
        return self.train_data()

    def val_data(self):
        return torch.utils.data.DataLoader(
            self.svg_loader(
                augmentations=self.val_augmentations,
                cfg=self.cfg,
                pre_load=True,
                source_list=self.cfg.dataset.files.val,
                target_list=self.cfg.dataset.files.mmd.val,
            ),
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=self.cfg.dataset.pin_memory,
            persistent_workers=self.cfg.dataset.persistent_workers,
        )
    
    def val_dataloader(self):
        return self.val_data()

    def test_dataloader(self, data_list=None, batch_size=1, pre_load=False):
        return torch.utils.data.DataLoader(
            FloorplanSVG(
                augmentations=DictToTensor(),
                cfg=self.cfg,
                pre_load=pre_load,
                source_list=data_list
            ),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=self.cfg.dataset.num_workers,
            persistent_workers=False
        )