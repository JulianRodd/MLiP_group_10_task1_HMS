
from datasets.data_loader import CustomDataset
from datasets.data_loader_configs import BaseConfig_Train_Annotator_Agreement, Config_Normalize_Group_Raw_ICA, Config_Normalize_Group_Raw_MSPCA, SmallBaseConfig_Train_Annotator_Agreement, SmallConfig_Normalize_Group_Raw_ICA, ExtremelySmallBaseConfig
from generics import Paths
from models.CustomModel import CustomModel
from models.custom_model_configs import BaseModelConfig, EfficientNetB0Config_Big, EfficientNetB0Config_Big_Weight_Decay_FROZEN_32, EfficientNetB0Config_Big_Weight_Decay_Only_Custom_spectrograms
from utils.data_preprocessing_utils import filter_by_agreement, filter_by_annotators
from datasets.raw_data_loader import CustomRawDataset
from utils.general_utils import get_logger
from utils.grid_search_utils import grid_search
from utils.inference_utils import perform_inference
from utils.loader_utils import load_main_dfs
from utils.training_utils import train

def main():
    logger = get_logger("main")
    
    data_loader_config = Config_Normalize_Group_Raw_ICA

    model_config = EfficientNetB0Config_Big_Weight_Decay_Only_Custom_spectrograms
    
    logger.info(f"Training model {model_config.NAME} with data loader {data_loader_config.NAME}")
    
    train_df, val_df, test_df = load_main_dfs(data_loader_config, train_val_split=(0.8, 0.2))
    
    # Load datasets
    train_dataset = CustomDataset(config=data_loader_config, main_df = train_df, mode="train", cache=True)
    val_dataset = CustomDataset(config=data_loader_config,main_df = val_df, mode="val", cache=True)
    

    # Print summaries
    train_dataset.print_summary()
    val_dataset.print_summary()


    # # Initialize and train the model
    # model = CustomModel(model_config)
    # train(model=model, train_dataset=train_dataset, val_dataset=val_dataset, tensorboard_prefix="agreement_test")
    
    # modelDir = f"{Paths.BEST_MODEL_CHECKPOINTS}/best_{model_config.MODEL}_{model_config.NAME}_{data_loader_config.NAME}.pth"
    
    # perform_inference(test_dataset, model, modelDir)
    # grid_search("Test different models", 3)

if __name__ == "__main__":
    main()