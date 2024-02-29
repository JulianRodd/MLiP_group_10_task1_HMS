class BaseModelConfig:
    NAME = "BaseModelConfig"
    KLDIV_REDUCTION = "batchmean"
    OPTIMIZER = "adam"
    WEIGHT_DECAY = 0
    LEARNING_RATE = 0.1
    MAX_LEARNING_RATE_SCHEDULERER = 0.01
    SCHEDULER = "CosineAnnealingLR"
    USE_KAGGLE_SPECTROGRAMS = True
    USE_EEG_SPECTROGRAMS = True
    GRADIENT_ACCUMULATION_STEPS = 1
    T_MAX = 10
    NUM_FROZEN_LAYERS = 32
    MAX_GRAD_NORM = 1000
    PRINT_FREQ = 10
    ETA_MIN = 0
    AMP = False
    # This method is called whenever a subclass of BaseModelConfig is created
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.NAME = cls.__name__


class ShuffleNetTest(BaseModelConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL = 'shufflenet_v2_x1_0'
    FREEZE = False
    EPOCHS = 10 

class ResNetBase(BaseModelConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL = 'resnet50'
    FREEZE = False
    EPOCHS = 10 

class EfficientNetB0Config_Big(BaseModelConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL = "tf_efficientnet_b0"
    FREEZE = False
    EPOCHS = 60

class EfficientNetB1Config_Big(BaseModelConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL = "tf_efficientnet_b1"
    FREEZE = False
    EPOCHS = 60

class EfficientNetB0Config_Big_Weight_Decay(BaseModelConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL = "tf_efficientnet_b0"
    FREEZE = False
    EPOCHS = 60
    WEIGHT_DECAY = 0.01
    
    
class EfficientNetB0Config_Big_Weight_Decay_Only_Custom_spectrograms(BaseModelConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL = "tf_efficientnet_b0"
    USE_KAGGLE_SPECTROGRAMS = False
    FREEZE = False
    EPOCHS = 60
    WEIGHT_DECAY = 0.01
    
class EfficientNetB0Config_Big_Weight_Decay_FROZEN_32(BaseModelConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL = "tf_efficientnet_b0"
    FREEZE = True
    NUM_FROZEN_LAYERS = 32
    EPOCHS = 60
    WEIGHT_DECAY = 0.01

class EfficientNetB1Config_Big_Weight_Decay(BaseModelConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL = "tf_efficientnet_b1"
    FREEZE = False
    EPOCHS = 60
    WEIGHT_DECAY = 0.01
    
    
  


MODEL_GRID_SEARCH = [EfficientNetB0Config_Big, EfficientNetB1Config_Big, EfficientNetB0Config_Big_Weight_Decay, EfficientNetB1Config_Big_Weight_Decay]