
from datasets.data_loader import CustomDataset
from datasets.data_loader_configs import FullDataCustomPreprocessingConfig
from generics import Paths
from models.CustomModel import CustomModel
from models.custom_model_configs import ShuffleNetBase
from utils.general_utils import get_logger
from utils.inference_utils import perform_inference
from utils.loader_utils import load_main_dfs
from utils.training_utils import train

# class FullDataConfig(BaseDataConfig):
#     SUBSET_SAMPLE_COUNT = 0
#     BATCH_SIZE_TEST = 16
#     BATCH_SIZE_VAL = 16
#     BATCH_SIZE_TRAIN = 32

# class FullDataCustomPreprocessingConfig(BaseDataConfig):
#     SUBSET_SAMPLE_COUNT = 0
#     BATCH_SIZE_TEST = 16
#     BATCH_SIZE_VAL = 16
#     BATCH_SIZE_TRAIN = 32
#     USE_PRELOADED_EEG_SPECTROGRAMS = False
#     USE_PRELOADED_SPECTROGRAMS = False
#     CONFIG = {
#         "sfreq": 200,
#         "l_freq": 1, 
#         "h_freq": 70,
#         "save": False
    # }


# class ShuffleNetBase(BaseModelConfig):
#     GRADIENT_ACCUMULATION_STEPS = 1
#     MODEL = 'shufflenet_v2_x1_0'
#     FREEZE = False
#     EPOCHS = 10
#     LEARNING_RATE = 0.01

# class ShuffleNetBaseLR01(BaseModelConfig):
#     GRADIENT_ACCUMULATION_STEPS = 1
#     MODEL = 'shufflenet_v2_x1_0'
#     FREEZE = False
#     EPOCHS = 50
#     LEARNING_RATE = 0.1

# class ShuffleNetBaseWD(BaseModelConfig):
#     GRADIENT_ACCUMULATION_STEPS = 1
#     MODEL = 'shufflenet_v2_x1_0'
#     FREEZE = False
#     EPOCHS = 50
#     LEARNING_RATE = 0.01
#     WEIGHT_DECAY = 0.01




def main():
    logger = get_logger("main")
    
    data_loader_config = FullDataCustomPreprocessingConfig

    model_config = ShuffleNetBase
    
    logger.info(f"Training model {model_config.NAME} with data loader {data_loader_config.NAME}")
    
    train_df, val_df, test_df = load_main_dfs(data_loader_config, train_val_split=(0.8, 0.2))
    
    # Load datasets
    train_dataset = CustomDataset(config=data_loader_config, main_df = train_df, mode="train", cache=True)
    val_dataset = CustomDataset(config=data_loader_config,main_df = val_df, mode="val", cache=True)
    
    # Print summaries
    train_dataset.print_summary()
    val_dataset.print_summary()

    # Initialize and train the model
    model = CustomModel(model_config)
    train(model=model, train_dataset=train_dataset, val_dataset=val_dataset, tensorboard_prefix="shufflenet_test_wd")
    
    modelDir = f"{Paths.BEST_MODEL_CHECKPOINTS}/best_{model_config.MODEL}_{model_config.NAME}_{data_loader_config.NAME}.pth"

    perform_inference(test_df, model, modelDir)
    # grid_search("Test different models", 3)

if __name__ == "__main__":
    main()