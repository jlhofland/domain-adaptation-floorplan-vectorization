from models.cubicasa import CubiCasa
from models.cubicasa21M import CubiCasa21M
from torch import nn


def factory(cfg):
    if cfg.model.name == 'CubiCasa' or cfg.model.name == 'CubiCasa21M':
        # Create the model and initialize weights
        model = eval(cfg.model.name)(51, cfg.mmd.latent_channels == 256, cfg.mmd.latent_transformation)

        # Initialize the weights of the model
        if cfg.model.init_weights and cfg.model.name != 'CubiCasa21M':
            model.init_weights()
            print("Weights initialized.")
        else:
            print("Warning: init_weights is not supported for CubiCasa21M due to different architecture.")

        # Count the number of classes
        n_classes = sum(cfg.model.input_slice)

        # Modify for greyscale input (1 channel instead of 3)
        if cfg.dataset.grayscale:
            model.conv1_ = nn.Conv2d(1, 64, bias=True, kernel_size=7, stride=2, padding=3)

        # Modify the model architecture for the specific task
        model.conv4_ = nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
        model.upsample = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)

        # Initialize the weights of the modified layers
        init_weights = [model.conv4_, model.upsample]

        # Initialize the weights of the modified layers
        for m in init_weights:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        # Return the model
        return model
    else:
        raise NotImplementedError(f"Model {cfg.model.name}")
