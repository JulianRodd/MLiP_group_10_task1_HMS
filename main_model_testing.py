from datasets.data_loader import CustomDataset
from datasets.data_loader_configs import BaseDataConfig, BaseLarge
from generics import Paths
from models.custom_model_configs import EffNetControl
from models.CustomModel import CustomModel
from utils.general_utils import get_logger
from utils.inference_utils import perform_inference
from utils.loader_utils import load_main_dfs
from utils.training_utils import train


class FullDataConfig(BaseDataConfig):
    SUBSET_SAMPLE_COUNT = 0
    BATCH_SIZE_TEST = 16
    BATCH_SIZE_VAL = 16
    BATCH_SIZE_TRAIN = 32
    USE_PRELOADED_EEG_SPECTROGRAMS = False
    USE_PRELOADED_SPECTROGRAMS = False


# This main is used to test the training of a model
def main():
    logger = get_logger("main")

    data_loader_config = BaseLarge

    model_config = EffNetControl

    logger.info(
        f"Training model {model_config.NAME} with data loader {data_loader_config.NAME}"
    )

    train_df, val_df, test_df = load_main_dfs(
        data_loader_config, train_val_split=(0.8, 0.2)
    )

    # Load datasets
    train_dataset = CustomDataset(
        config=data_loader_config, main_df=train_df, mode="train", cache=False
    )
    val_dataset = CustomDataset(
        config=data_loader_config, main_df=val_df, mode="val", cache=False
    )

    # Print summaries
    train_dataset.print_summary()
    val_dataset.print_summary()

    # Initialize and train the model
    model = CustomModel(model_config)
    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tensorboard_prefix="BUG_HUNT_EFFNET_GOOD_VAL",
    )

    modelDir = f"{Paths.BEST_MODEL_CHECKPOINTS}/best_{model_config.MODEL}_{model_config.NAME}_{data_loader_config.NAME}.pth"

    perform_inference(test_df, model, modelDir)
    # grid_search("Test different models", 3)


if __name__ == "__main__":
    main()
