from multiprocessing import get_logger
from datasets.data_loader import CustomDataset
from datasets.data_loader_configs import Config_Normalize_Group_Raw_ICA
from generics import Paths
from models.CustomModel import CustomModel
from models.custom_model_configs import EfficientNetB0Config_Big
from utils.inference_utils import perform_inference
from utils.loader_utils import load_main_dfs
from utils.training_utils import train

def main():
    logger = get_logger("main")
    
    data_loader_config = Config_Normalize_Group_Raw_ICA
    model_config = EfficientNetB0Config_Big
    
    logger.info(f"Training model {model_config.NAME} with data loader {data_loader_config.NAME}")
      
    train_df, val_df, test_df = load_main_dfs(data_loader_config.SUBSET_SAMPLE_COUNT, train_val_split=(0.8, 0.2))
    # Load datasets
    train_dataset = CustomDataset(config=data_loader_config, main_df = train_df, mode="train", cache=False)
    val_dataset = CustomDataset(config=data_loader_config,main_df = val_df, mode="val", cache=False)
    test_dataset = CustomDataset(config=data_loader_config,main_df = test_df, mode="test", cache=False)

    # Print summaries
    train_dataset.print_summary()
    val_dataset.print_summary()
    test_dataset.print_summary()

    # Initialize and train the model
    model = CustomModel(model_config)
    train(model=model, train_dataset=train_dataset, val_dataset=val_dataset)
    
    modelDir = f"{Paths.BEST_MODEL_CHECKPOINTS}/best_{model_config.MODEL}_{model_config.NAME}_{data_loader_config.NAME}.pth"
    
    perform_inference(test_dataset, model, modelDir)

if __name__ == "__main__":
    main()