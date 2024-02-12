class BaseConfig:
    ONE_CROP_PER_PERSON = True
    USE_WAVELET: bool = None
    BATCH_SIZE_TRAIN = 64
    BATCH_SIZE_TEST = 16
    BATCH_SIZE_VAL = 16
    SHUFFLE_TRAIN = True
    NUM_WORKERS = 4
    PIN_MEMORY = True
    DROP_LAST = True
    NAMES = ["LL", "LP", "RP", "RR"]
    FEATS = [
        ["Fp1", "F7", "T3", "T5", "O1"],
        ["Fp1", "F3", "C3", "P3", "O1"],
        ["Fp2", "F8", "T4", "T6", "O2"],
        ["Fp2", "F4", "C4", "P4", "O2"],
    ]
    VAL_SPLIT_RATIO = 0.2
    SUBSET_SAMPLE_COUNT: int = 0


# Extend BaseConfig to change or dataset configuration
class ExtremelySmallBaseConfig(BaseConfig):
    BATCH_SIZE_TRAIN = 2
    BATCH_SIZE_TEST = 2
    BATCH_SIZE_VAL = 1
    SUBSET_SAMPLE_COUNT: int = 10
