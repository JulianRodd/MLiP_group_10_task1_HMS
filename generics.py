import torch


# This file contains all the generic variables and paths used throughout the project
class Paths:
    PRE_LOADED_EEGS = "./data/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy"
    PRE_LOADED_CUSTOM_EEGS_DIR = "./data/cache/"
    PRE_LOADED_SPECTROGRAMS = "./data/kaggle/input/brain-spectrograms/specs.npy"
    TRAIN_CSV = (
        "./data/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
    TEST_CSV = "./data/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = (
        "./data/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    )
    TRAIN_EEGS = (
        "./data/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    )
    TRAIN_SPECTROGRAMS = "./data/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    TEST_SPECTROGRAMS = "./data/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"

    OTHER_MODEL_CHECKPOINTS = "./checkpoints/other_models/"
    BEST_MODEL_CHECKPOINTS = "./checkpoints/best_models/"
    CACHE_PATH_READ = "./data/cache/"
    CACHE_PATH_WRITE = "./data/cache/"
    TENSORBOARD = "./tensorboard/"
    TENSORBOARD_MODELS = "./tensorboard/models/"
    TENSORBOARD_TRAINING = "./tensorboard/training/"
    TENSORBOARD_DATASETS = "./tensorboard/datasets/"
    TENSORBOARD_INFERENCE = "./tensorboard/inference/"

    LOG_PATH = "./logs/"


# This class contains all the generic variables used throughout the project
class Generics:
    OPT_OUT_EEG_ID = []
    LABEL_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    TARGET_PREDS = [x + "_pred" for x in LABEL_COLS]
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
