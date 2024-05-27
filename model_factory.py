from models.cubicasa import CubiCasa
from torch import nn


def factory(cfg):
    if cfg.model.name == 'CubiCasa':
        # Create the model and initialize weights
        model = CubiCasa(51)
        model.init_weights()

        # Count the number of classes
        n_classes = sum(cfg.model.input_slice)

        # Modify for greyscale input (1 channel instead of 3)
        if cfg.dataset.grayscale:
            model.conv1_ = nn.Conv2d(1, 64, bias=True, kernel_size=7, stride=2, padding=3)

        # Modify the model architecture for the specific task
        model.conv4_ = nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
        model.upsample = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)

        init_weights = [model.conv4_, model.upsample]

        # # Create fully connected layer to get latent representation
        # if cfg.model.use_mmd:
        #     # [batch_size, 512, H/64, W/64] --> [batch_size, 1, H/64, W/64]
        #     model.reduce_conv = nn.Conv2d(512, 1, kernel_size=1)
        #     init_weights.append(model.reduce_conv)

        # Initialize the weights of the modified layers
        for m in init_weights:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        # Return the model
        return model
    else:
        raise NotImplementedError(f"Model {cfg.model.name}")
