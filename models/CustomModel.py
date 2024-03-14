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
        tensorboard_prefix: str = "all",
        torch_model_cache_dir: str = "pytorch/vision:v0.10.0",
    ):
        super(CustomModel, self).__init__()
        self.logger = get_logger(f"models/{__name__}")
        self.logger.info(f"Using device: {device}")
        self.device = device

        self.num_classes = num_classes
        self.config = config
        self.model = (
            timm.create_model(
                config.MODEL,
                pretrained=pretrained,
                drop_rate=0.1,
                drop_path_rate=0.2,
            )
            if config.MODEL.startswith("tf_")
            else torch.hub.load(
                torch_model_cache_dir,
                config.MODEL,
                pretrained=pretrained,
                source=(
                    "github"
                    if torch_model_cache_dir == "pytorch/vision:v0.10.0"
                    else "local"
                ),
            )
        )
        if config.FREEZE:
            for i, (name, param) in enumerate(
                list(self.model.named_parameters())[0 : config.NUM_FROZEN_LAYERS]
            ):
                param.requires_grad = False

        self.features = self.set_feature_layers()
        self.custom_layers = self.set_custom_layers()
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                Paths.TENSORBOARD_MODELS, f"{tensorboard_prefix}/{config.NAME}"
            )
        )
        self.to(self.device)
        self.logger.info(f"{config.MODEL} initialized with config {config.NAME}")

    def set_custom_layers(self):
        # this should probs become a dict once we know which sizes we are going to use
        if self.config.MODEL.startswith("tf_"):
            num_features = self.model.num_features
        elif self.config.MODEL.startswith("shufflenet"):
            num_features = 1024  # need to make this better
        elif self.config.MODEL.startswith("resnet"):
            num_features = 2048
        else:
            raise NotImplementedError("Model not implemented - check model name.")

        if getattr(
            self.config, "LARGE_CLASSIFIER", False
        ):  # not all will have attribute so to not break it return False if attr does not exist
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(num_features, 256),
                nn.BatchNorm1d(
                    256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
                ),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.num_classes),
            )

        else:
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(num_features, self.num_classes),
            )

    def set_feature_layers(self):
        if self.config.MODEL.startswith("tf_") or self.config.MODEL.startswith(
            "resnet"
        ):
            return nn.Sequential(*list(self.model.children())[:-2])

        elif self.config.MODEL.startswith("shufflenet"):
            return nn.Sequential(*list(self.model.children())[:-1])

    def log_model_parameters(self, step: int):
        """
        Logs the model parameters to TensorBoard.
        """
        for name, param in self.named_parameters():
            self.writer.add_histogram(f"{name.replace('.', '/')}", param, step)
            if param.grad is not None:
                self.writer.add_histogram(
                    f"{name.replace('.', '/')}/grad", param.grad, step
                )

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
        if self.config.USE_KAGGLE_SPECTROGRAMS & self.config.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spectograms, eegs], dim=2)
        elif self.config.USE_EEG_SPECTROGRAMS:
            x = eegs
        else:
            x = spectograms

        x = torch.cat([x, x, x], dim=3)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x
