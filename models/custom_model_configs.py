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
    AMP = False
    # This method is called whenever a subclass of BaseModelConfig is created
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.NAME = cls.__name__


class EfficientNetB0Config_Big(BaseModelConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL = "efficientnet_b0"
    FREEZE = False
    EPOCHS = 30

class EfficientNetB0Config(BaseModelConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL = "efficientnet_b0"
    FREEZE = False
    EPOCHS = 4
    
class EfficientNetB1Config(BaseModelConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL = "efficientnet_b1"
    FREEZE = False
    EPOCHS = 4


class EfficientNetB0ConfigV1(EfficientNetB0Config):
    EPOCHS = 10
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 1e-5

class EfficientNetB0ConfigV2(EfficientNetB0Config):
    FREEZE = True
    EPOCHS = 6
    T_MAX = 5

class EfficientNetB0ConfigV3(EfficientNetB0Config):
    GRADIENT_ACCUMULATION_STEPS = 2
    EPOCHS = 8
    MAX_GRAD_NORM = 500

class EfficientNetB1ConfigV1(EfficientNetB1Config):
    EPOCHS = 5
    LEARNING_RATE = 0.002

class EfficientNetB1ConfigV2(EfficientNetB1Config):
    FREEZE = True
    EPOCHS = 3
    ETA_MIN = 0.0001

class EfficientNetB1ConfigV3(EfficientNetB1Config):
    GRADIENT_ACCUMULATION_STEPS = 2
    EPOCHS = 7
    MAX_GRAD_NORM = 800

# Following are additional variations
class EfficientNetB0ConfigV4(EfficientNetB0Config):
    EPOCHS = 5

class EfficientNetB0ConfigV5(EfficientNetB0Config):
    FREEZE = True
    EPOCHS = 9

class EfficientNetB1ConfigV4(EfficientNetB1Config):
    EPOCHS = 2

class EfficientNetB1ConfigV5(EfficientNetB1Config):
    FREEZE = False
    EPOCHS = 12
    
class EfficientNetB0ConfigV1_Small(EfficientNetB0ConfigV1):
    EPOCHS = 5
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-6
    
MODEL_GRID_SEARCH = [EfficientNetB0ConfigV1_Small]