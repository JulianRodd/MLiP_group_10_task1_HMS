class BaseModelConfig:
    NAME = "BaseModelConfig"
    KLDIV_REDUCTION = "batchmean"
    OPTIMIZER = "adam"
    WEIGHT_DECAY = 0
    LEARNING_RATE = 0.1
    MAX_LEARNING_RATE_SCHEDULERER = 0.001
    SCHEDULER = "CosineAnnealingLR"
    USE_KAGGLE_SPECTROGRAMS = True
    USE_EEG_SPECTROGRAMS = True
    GRADIENT_ACCUMULATION_STEPS = 1
    T_MAX = 10
    NUM_FROZEN_LAYERS = 32
    MAX_GRAD_NORM = 1e7
    PRINT_FREQ = 10
    ETA_MIN = 0
    AMP = True

    # This method is called whenever a subclass of BaseModelConfig is created
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.NAME = cls.__name__


class EffNetControl(BaseModelConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL = "tf_efficientnet_b0"
    FREEZE = False
    EPOCHS = 4
    LARGE_CLASSIFIER = False
    WEIGHT_DECAY = 0.01
    AMP = True
    MAX_GRAD_NORM = 1e7


class ShuffleNetBase(BaseModelConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL = "shufflenet_v2_x1_0"
    FREEZE = False
    EPOCHS = 20
    LEARNING_RATE = 0.01


class ShuffleNetBase_Large(BaseModelConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL = "shufflenet_v2_x1_0"
    FREEZE = False
    EPOCHS = 20
    LEARNING_RATE = 0.01
    LARGE_CLASSIFIER = True


class ShuffleNetTest(BaseModelConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL = "shufflenet_v2_x1_0"
    FREEZE = False
    EPOCHS = 10


class ResNetBase(BaseModelConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL = "resnet50"
    FREEZE = False
    EPOCHS = 10


class ResNetBase_LargeCF(BaseModelConfig):
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL = "resnet50"
    FREEZE = False
    EPOCHS = 10
    LARGE_CLASSIFIER = True
    WEIGHT_DECAY = 0.01


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


MODEL_SEARCH = [
    EfficientNetB0Config_Big,
    EfficientNetB1Config_Big,
    EfficientNetB0Config_Big_Weight_Decay,
    EfficientNetB1Config_Big_Weight_Decay,
]
