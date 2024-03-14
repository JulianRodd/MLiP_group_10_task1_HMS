import pandas as pd
from datasets.data_loader import CustomDataset
from generics import Generics, Paths
from models.CustomModel import CustomModel
from utils.general_utils import get_logger
from utils.inference_utils import create_submission, perform_inference
from utils.training_utils import perform_cross_validation, train_loop, get_result
from torch import nn
import torch
from datasets.data_loader_configs import ExtremelySmallBaseConfig
from models.custom_model_configs import EfficientNetB0Config

def main():
    logger = get_logger("main")
    data_loader_config = ExtremelySmallBaseConfig()
    
    # Load datasets
    train_dataset = CustomDataset(config=data_loader_config, mode="train", cache=True)
    val_dataset = CustomDataset(config=data_loader_config, mode="val", cache=True)
    test_dataset = CustomDataset(config=data_loader_config, mode="test", cache=True)
    
    # Print summaries
    train_dataset.print_summary()
    val_dataset.print_summary()
    test_dataset.print_summary()

    # Initialize and train the model
    model_config = EfficientNetB0Config()
    model = CustomModel(model_config)
    perform_cross_validation(model=model, train_dataset=train_dataset, val_dataset=val_dataset)
    
    # Define model directories for inference
    model_dirs = [f"{Paths.BEST_MODEL_CHECKPOINTS}/best_{model_config.MODEL}_{model_config.NAME}_{data_loader_config.NAME}_fold_{fold}.pth" for fold in range(model_config.FOLDS)]

    # Perform inference using the trained models
    predictions = perform_inference(test_dataset, model, model_dirs)

    # Create and save the submission file
    submission_df = create_submission(test_dataset.main_df, predictions, Generics.LABEL_COLS, 'submission.csv')
    print(submission_df.head())

if __name__ == "__main__":
    main()