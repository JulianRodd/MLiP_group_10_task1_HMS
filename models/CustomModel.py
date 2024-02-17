import os
import timm
import torch
import torch.nn as nn
from generics import Generics, Paths
from utils.general_utils import get_logger
from torch.utils.tensorboard import SummaryWriter

class CustomModel(nn.Module):
    def __init__(
        self,
        config,
        num_classes: int = 6,
        pretrained: bool = True,
        device=Generics.DEVICE,
    ):
        super(CustomModel, self).__init__()
        self.logger = get_logger(f"models/{__name__}")
        self.logger.info(f"Using device: {device}")
        self.device = device
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.config = config
        self.model = timm.create_model(
            config.MODEL,
            pretrained=pretrained,
            drop_rate=0.1,
            drop_path_rate=0.2,
        )
        if config.FREEZE:
            for i, (name, param) in enumerate(
                list(self.model.named_parameters())[0 : config.NUM_FROZEN_LAYERS]
            ):
                param.requires_grad = False

        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes),
        )
        self.writer = SummaryWriter(log_dir=os.path.join(Paths.TENSORBOARD_MODELS, config.NAME))
        self.to(self.device)
        self.logger.info(f"{config.MODEL} initialized with config {config.NAME}")
      

    def log_model_parameters(self, step: int):
        """
        Logs the model parameters to TensorBoard.
        """
        for name, param in self.named_parameters():
            self.writer.add_histogram(f"{name.replace('.', '/')}", param, step)
            if param.grad is not None:
                self.writer.add_histogram(f"{name.replace('.', '/')}/grad", param.grad, step)
     
                  
    def __reshape_input(self, x):
        """
        Reshapes input (128, 256, 8) -> (512, 512, 3) monotone image.
        """
        # === Get spectograms ===
        spectograms = [x[:, :, :, i : i + 1] for i in range(4)]
        spectograms = torch.cat(spectograms, dim=1)

        # === Get EEG spectograms ===
        eegs = [x[:, :, :, i : i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=1)

        # === Reshape (512,512,3) ===
        if self.USE_KAGGLE_SPECTROGRAMS & self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spectograms, eegs], dim=2)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eegs
        else:
            x = spectograms

        x = torch.cat([x, x, x], dim=3)
        x = x.permute(0, 3, 1, 2)
        x = x.to(self.device).float()
        return x

    def forward(self, x):
        x = x.to(self.device).float()  # Ensure x is a float tensor
        x = self.__reshape_input(x)
        x = x.to(self.device).float()
        x = self.features(x)
        x = self.custom_layers(x)
        return x
