from torch import nn
import torch

class BaseConfig:
  KLDIV_REDUCTION = "batchmean"
  OPTIMIZER = "adam"
  LEARNING_RATE = 0.001
  SCHEDULER = "CosineAnnealingLR"
  T_MAX = 10
  MAX_GRAD_NORM = 1000
  PRINT_FREQ = 10

  
class EfficientNetB0Config(BaseConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    AMP = False
    MODEL = "efficientnet_b0"
    FREEZE = False
    FOLDS = 5
    EPOCHS = 2
