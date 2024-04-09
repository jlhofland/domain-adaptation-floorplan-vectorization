"""
PyTorch Lightning example code, designed for use in TU Delft CV lab.

Copyright (c) 2022 Robert-Jan Bruintjes, TU Delft.
"""
# Package imports, from conda or pip
import os
import warnings
import torch
from omegaconf import OmegaConf

# PyTorch Lightning
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.tuner import Tuner
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

# Datasets
from datasets.cubicasa5k.runner import Runner
from datasets.cubicasa5k.data_loader import CubiCasa

# Import factories
import model_factory
import loss_factory


def main():
    # Remove UserWarning related to libpng
    warnings.filterwarnings("ignore", category=UserWarning, module=".*libpng.*")

    # Load defaults and overwrite by command-line arguments
    cfg = OmegaConf.load("config.yaml")
    cmd_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cmd_cfg)
    print(OmegaConf.to_yaml(cfg))

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Seed everything. Note that this does not make training entirely deterministic
    seed_everything(cfg.seed, workers=True)

    # Set cache dir to W&B logging directory
    os.environ["WANDB_CACHE_DIR"] = os.path.join(cfg.wandb.dir, 'cache')
    wandb_logger = WandbLogger(
        save_dir=cfg.wandb.dir,
        project=cfg.wandb.project,
        name=cfg.wandb.experiment_name,
        log_model='all' if cfg.wandb.log else None,
        offline=not cfg.wandb.log,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_object(cfg),
    )

    # Add labels to the logger
    labels = {
        "loss": ['all', 'rooms', 'icons', 'heatmap', 'all_var', 'rooms_var', 'icons_var', 'heatmap_var'],
        "room": ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"],
        "icon": ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"],
        "heat": ["I-UP", "I-RIGHT", "I-DOWN", "I-LEFT", # I junctions
                 "L-UP-RIGHT", "L-RIGHT-DOWN", "L-DOWN-LEFT", "L-LEFT-UP", # L junctions
                 "T-LEFT-UP-RIGHT", "T-UP-RIGHT-DOWN", "T-RIGHT-DOWN-LEFT", "T-DOWN-LEFT-UP", # T junctions
                 "X-UP-RIGHT-DOWN-LEFT", # X junction
                 "O-UP", "O-RIGHT", "O-DOWN", "O-LEFT", # O(pening) junctions
                 "ICON-UP-RIGHT", "ICON-RIGHT-DOWN", "ICON-DOWN-LEFT", "ICON-LEFT-UP"] # ICON junctions
    }

    # Create loss function
    loss_fn = loss_factory.factory(cfg)

    # Create model using factory pattern
    model = model_factory.factory(cfg)
    
    # Create datasets using factory pattern
    dataset = CubiCasa(cfg)

    # Set float32 precision for matrix multiplication (speedup on modern GPUs)
    torch.set_float32_matmul_precision("medium")

    # Watch model for logging, gradients, parameters, and optimizer parameters
    wandb_logger.watch(model, log="all")

    # Runner is the PyTorch Lightning module that contains the instructions for training and validation
    if cfg.model.weights and os.path.exists(cfg.model.weights):
        runner = Runner.load_from_checkpoint(cfg.model.weights, cfg=cfg, model=model, loss_fn=loss_fn, labels=labels)
    else:
        runner = Runner(cfg, model, loss_fn, labels)

    # Trainer executes the training/validation loops and model checkpointing.
    trainer = Trainer(
        # Timing
        max_epochs=cfg.train.max_epochs,
        max_time={
            "days": cfg.train.max_time.days, 
            "hours": cfg.train.max_time.hours, 
            "minutes": cfg.train.max_time.minutes
        } if cfg.train.max_time.enable else None,

        # Speedup 
        strategy="auto", 
        accelerator="auto",
        num_nodes=torch.cuda.device_count(),
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        precision="32-true", 
        limit_train_batches=cfg.train.debug.train_batches if cfg.train.debug.enable else None,
        limit_val_batches=cfg.train.debug.val_batches if cfg.train.debug.enable else None,
        limit_test_batches=cfg.train.debug.test_batches if cfg.train.debug.enable else None,

        # Logging and debugging
        detect_anomaly=cfg.debugger.anomaly,
        logger=wandb_logger,
        fast_dev_run=cfg.debugger.batches,
        profiler=eval(cfg.debugger.profiler)(dirpath=cfg.wandb.dir+"/profiler", filename=cfg.wandb.experiment_name) if cfg.debugger.profiler else None,
        callbacks=[DeviceStatsMonitor(cpu_stats=True)] if cfg.debugger.accelerator else None
    )

    # Scale batch size using PyTorch Lightning's Tuner
    if cfg.debugger.batch_tuner:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(runner, datamodule=dataset)

    # # Train and validate from scratch or resume from checkpoint
    if cfg.train.resume and os.path.exists(cfg.train.resume):
        trainer.fit(runner, datamodule=dataset, ckpt_path=cfg.train.resume)
    else:
        trainer.fit(runner, datamodule=dataset)

    # Test
    trainer.test(runner, dataset.test_dataloader())

if __name__ == '__main__':
    # Ignore UserWarning related to libpng
    warnings.filterwarnings("ignore", category=UserWarning, module=".*libpng.*")
    main()