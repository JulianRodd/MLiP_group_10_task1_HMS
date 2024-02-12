class Paths:
    CACHE_DIR = "./data/cache/"
    PRE_LOADED_EEGS = "./data/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy"
    PRE_LOADED_SPECTROGRAMS = "./data/kaggle/input/brain-spectrograms/specs.npy"
    TRAIN_CSV = (
        "./data/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
    TEST_CSV = "./data/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "./data/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TRAIN_EEGS = "./data/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    TRAIN_SPECTROGRAMS = "./data/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    TEST_SPECTROGRAMS = "./data/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"

    OTHER_MODEL_CHECKPOINTS = "./checkpoints/other_models/"
    BEST_MODEL_CHECKPOINTS = "./checkpoints/best_models/"

    CACHE_PATH = "./data/cache/"
    TENSORBOARD = "./tensorboard/"
    
    LOG_PATH = "./logs/"


class EEGConfig:
    USE_WAVELET: bool = None
    NAMES = ["LL", "LP", "RP", "RR"]
    FEATS = [
        ["Fp1", "F7", "T3", "T5", "O1"],
        ["Fp1", "F3", "C3", "P3", "O1"],
        ["Fp2", "F8", "T4", "T6", "O2"],
        ["Fp2", "F4", "C4", "P4", "O2"],
    ]


class DataConfig:
    ONE_CROP_PER_PERSON = True
    USE_WAVELET: bool = None
    NAMES = ["LL", "LP", "RP", "RR"]
    FEATS = [
        ["Fp1", "F7", "T3", "T5", "O1"],
        ["Fp1", "F3", "C3", "P3", "O1"],
        ["Fp2", "F8", "T4", "T6", "O2"],
        ["Fp2", "F4", "C4", "P4", "O2"],
    ]
    EKG_FEAT = "EKG"


class Generics:
    OPT_OUT_EEG_ID = [1457334423]
