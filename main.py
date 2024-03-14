from datasets.data_loader import CustomDataset
from datasets.data_loader_configs import BaseDataConfig
from utils.loader_utils import load_main_dfs


class FullDataCustomPreprocessingConfig(BaseDataConfig):
    SUBSET_SAMPLE_COUNT = 0
    BATCH_SIZE_TEST = 16
    BATCH_SIZE_VAL = 16
    BATCH_SIZE_TRAIN = 32
    USE_PRELOADED_EEG_SPECTROGRAMS = False
    USE_PRELOADED_SPECTROGRAMS = False
    CONFIG = {"sfreq": 200, "l_freq": 1, "h_freq": 70, "save": True}


# This is a simple example used to test loading and preprocessing of the data
def main():
    data_loader_config = FullDataCustomPreprocessingConfig

    train_df, _, _ = load_main_dfs(data_loader_config, train_val_split=(0.8, 0.2))

    # Load datasets
    train_dataset = CustomDataset(
        config=data_loader_config, main_df=train_df, mode="train", cache=True
    )


if __name__ == "__main__":
    main()
