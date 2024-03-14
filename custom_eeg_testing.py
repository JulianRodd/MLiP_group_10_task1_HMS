import pandas as pd
from datasets.data_loader import CustomDataset
from datasets.data_loader_configs import BaseDataConfig
from generics import Paths, Generics
from models.CustomModel import CustomModel
from models.custom_model_configs import ShuffleNetBase
from utils.general_utils import get_logger
from utils.inference_utils import perform_inference, create_submission
from utils.loader_utils import load_main_dfs


class TestPreprocessingConfig(BaseDataConfig):
    SUBSET_SAMPLE_COUNT = 10
    BATCH_SIZE_TEST = 16
    BATCH_SIZE_VAL = 16
    BATCH_SIZE_TRAIN = 32
    USE_PRELOADED_EEG_SPECTROGRAMS = False
    USE_PRELOADED_SPECTROGRAMS = False
    DROP_LAST = False
    CONFIG = {
        "sfreq": 200,
        "l_freq": 1, 
        "h_freq": 70,
        "save": True
    }

class PreprocessingConfig(BaseDataConfig):
    SUBSET_SAMPLE_COUNT = 0
    BATCH_SIZE_TEST = 16
    BATCH_SIZE_VAL = 16
    BATCH_SIZE_TRAIN = 32
    USE_PRELOADED_EEG_SPECTROGRAMS = False
    USE_PRELOADED_SPECTROGRAMS = False
    DROP_LAST = False
    CONFIG = {
        "sfreq": 200,
        "l_freq": 1, 
        "h_freq": 70,
        "save": True
    }


class TestPreprocessingInferenceConfig(BaseDataConfig):
    SUBSET_SAMPLE_COUNT = 0
    BATCH_SIZE_TEST = 16
    BATCH_SIZE_VAL = 16
    BATCH_SIZE_TRAIN = 32
    USE_PRELOADED_EEG_SPECTROGRAMS = False
    USE_PRELOADED_SPECTROGRAMS = False
    DROP_LAST = False
    CONFIG = {
        "sfreq": 200,
        "l_freq": 1, 
        "h_freq": 70,
        "save": True
    }


def main():
    data_loader_config_train = PreprocessingConfig
    # data_loader_config_test = TestPreprocessingInferenceConfig
    custom_preprocessing_config = TestPreprocessingConfig.CONFIG

    train_df, val_df, test_df = load_main_dfs(data_loader_config_train, train_val_split=(0.8, 0.2))
    test_df = pd.read_csv(Paths.TEST_CSV)
    
    # Load datasets
    train_dataset = CustomDataset(config=data_loader_config_train, main_df = train_df, mode="train", cache=False, custom_preprocessing_config=custom_preprocessing_config)
    # train_dataset = CustomDataset(config=data_loader_config_train, main_df = train_df, mode="train", cache=False, custom_preprocessing_config=None)
    # val_dataset = CustomDataset(config=data_loader_config_train, main_df = val_df, mode="val", cache=False)

    # Load dataset
    # test_dataset = CustomDataset(config=data_loader_config_test, main_df = test_df, mode="test", cache=False)


if __name__ == "__main__":
    main()