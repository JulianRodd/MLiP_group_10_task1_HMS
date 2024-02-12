import os
import torch

class Paths:
    CACHE_DIR = "./data/cache/"
    PRE_LOADED_EEGS = "./data/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy"
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

    CACHE_PATH = "./data/cache/"
    TENSORBOARD = "./tensorboard/"

    LOG_PATH = "./logs/"




class Generics:
    OPT_OUT_EEG_ID = [1457334423]
    LABEL_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if os.environ.get("CUDA_MPS_PIPE_DIRECTORY") else "mps"
    )
