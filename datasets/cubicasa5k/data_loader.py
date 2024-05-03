import torch
import torchvision
import pytorch_lightning as pl

# Import cubicasa5k dataset
from datasets.cubicasa5k.loaders.svg_loader import FloorplanSVG
from datasets.cubicasa5k.loaders.augmentations import (RandomCropToSizeTorch, ResizePaddedTorch, Compose, DictToTensor, ColorJitterTorch, RandomRotations)
from torchvision.transforms import RandomChoice

class CubiCasa(pl.LightningDataModule):
    def __init__(self, cfg):
        # Call super constructor
        super().__init__()

        # Set batch size and number of workers
        self.batch_size = cfg.train.batch_size
        self.num_workers = cfg.dataset.num_workers
        self.format = cfg.dataset.format
        self.persistent_workers = cfg.dataset.persistent_workers
        self.original_size = cfg.dataset.original_size
        self.grayscale = cfg.dataset.grayscale
        self.load_ram = cfg.dataset.load_ram
        self.save_samples = cfg.dataset.save_samples
        self.load_samples = cfg.dataset.load_samples
        self.pin_memory = cfg.dataset.pin_memory

        # Scale or crop input
        size = (cfg.dataset.image_size, cfg.dataset.image_size)
        scale_augmentations = RandomChoice([
            RandomCropToSizeTorch(data_format='dict', size=size),
            ResizePaddedTorch((0, 0), data_format='dict', size=size)
        ]) if cfg.dataset.scale else RandomCropToSizeTorch(data_format='dict', size=size)

        # Set augmentations if not provided
        training_augmentations = Compose([
            scale_augmentations,
            RandomRotations(format='cubi'),
            DictToTensor(),
            ColorJitterTorch(gray=cfg.dataset.grayscale)
        ]) if cfg.dataset.augmentations else DictToTensor()

        # Set augmentations if provided
        self.augmentations = {
            "train": training_augmentations, 
            "val": DictToTensor(), 
            "test": DictToTensor()}

        # Set data root and files
        self.data_root = cfg.dataset.files.root
        self.train_file = cfg.dataset.files.train
        self.val_file = cfg.dataset.files.val
        self.test_file = cfg.dataset.files.test

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            FloorplanSVG(
                data_folder=self.data_root,
                data_file=self.train_file,
                is_transform=True,
                augmentations=self.augmentations['train'],
                img_norm=True,
                format=self.format,
                original_size=self.original_size,
                lmdb_folder='cubi_lmdb/',
                grayscale=self.grayscale, 
                load_ram=self.load_ram, 
                save_samples=self.save_samples,
                load_samples=self.load_samples,
                num_workers=self.num_workers,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            FloorplanSVG(
                data_folder=self.data_root,
                data_file=self.val_file,
                is_transform=True,
                augmentations=self.augmentations['val'],
                img_norm=True,
                format=self.format,
                original_size=self.original_size,
                lmdb_folder='cubi_lmdb/',
                grayscale=self.grayscale, 
                load_ram=self.load_ram, 
                save_samples=self.save_samples,
                load_samples=self.load_samples,
                num_workers=self.num_workers,
            ),
            batch_size=1,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            FloorplanSVG(
                data_folder=self.data_root,
                data_file=self.test_file,
                is_transform=True,
                augmentations=self.augmentations['test'],
                img_norm=True,
                format=self.format,
                original_size=self.original_size,
                lmdb_folder='cubi_lmdb/',
                grayscale=self.grayscale, 
                load_ram=False, 
                save_samples=False,
                load_samples=False,
                num_workers=self.num_workers,
            ),
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )