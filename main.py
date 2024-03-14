
from datasets.data_loader import CustomDataset
from datasets.data_loader_configs import BaseDataConfig
from generics import Paths
from models.CustomModel import CustomModel
from models.custom_model_configs import EffNetControl
from utils.general_utils import get_logger
from utils.inference_utils import perform_inference
from utils.loader_utils import load_main_dfs
from utils.training_utils import train


class FullDataCustomPreprocessingConfig(BaseDataConfig):
    SUBSET_SAMPLE_COUNT = 0
    BATCH_SIZE_TEST = 16
    BATCH_SIZE_VAL = 16
    BATCH_SIZE_TRAIN = 32
    USE_PRELOADED_EEG_SPECTROGRAMS = False
    USE_PRELOADED_SPECTROGRAMS = False
    CONFIG = {
        "sfreq": 200,
        "l_freq": 1, 
        "h_freq": 70,
        "save": True
    }




def main():
    logger = get_logger("main")
    
    data_loader_config = FullDataCustomPreprocessingConfig
    
    train_df, val_df, test_df = load_main_dfs(data_loader_config, train_val_split=(0.8, 0.2))
    
    # Load datasets
    train_dataset = CustomDataset(config=data_loader_config, main_df = train_df, mode="train", cache=True)
    

if __name__ == "__main__":
    main()