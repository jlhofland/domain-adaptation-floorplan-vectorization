import os
import pickle
from omegaconf import OmegaConf
from datasets.cubicasa5k.loaders.svg_loader import FloorplanSVG

files = ['train.txt', 'val.txt', 'test.txt']

# Load defaults and overwrite by command-line arguments
cfg = OmegaConf.load("experiments/config_pkl.yaml")
cmd_cfg = OmegaConf.from_cli()
cfg = OmegaConf.merge(cfg, cmd_cfg)
print(OmegaConf.to_yaml(cfg))

for file_txt in files:
    data = FloorplanSVG(file_txt, cfg, pre_load=True)
