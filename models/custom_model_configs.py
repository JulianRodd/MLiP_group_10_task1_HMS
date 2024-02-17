from torch import nn
import torch

class BaseModelConfig:
  NAME = "BaseModelConfig"
  KLDIV_REDUCTION = "batchmean"
  OPTIMIZER = "adam"
  WEIGHT_DECAY = 0
  LEARNING_RATE = 0.001
  SCHEDULER = "CosineAnnealingLR"
  T_MAX = 10
  MAX_GRAD_NORM = 1000
  PRINT_FREQ = 10
  ETA_MIN = 0

  
class EfficientNetB0Config(BaseModelConfig):
    NAME = "EfficientNetB0Config"
    GRADIENT_ACCUMULATION_STEPS = 1
    AMP = False
    MODEL = "efficientnet_b0"
    FREEZE = False
    FOLDS = 1
    EPOCHS = 1
