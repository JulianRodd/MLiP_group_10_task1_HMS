import multiprocessing
from functools import partial
import pandas as pd
from datasets.data_loader import CustomDataset
from generics import Paths
from models.CustomModel import CustomModel
from utils.general_utils import get_logger
from utils.inference_utils import perform_inference
from utils.training_utils import perform_cross_validation
from datasets.data_loader_configs import DATASET_GRID_SEARCH
from models.custom_model_configs import MODEL_GRID_SEARCH

def process_model_data_combo(data_loader_config, model_config):
    # Initialize the logger within each process
    logger = get_logger("grid_search_process")

    try:
        logger.info(f"Training model {model_config.NAME} with data loader {data_loader_config.NAME}")
        
        # Load datasets
        train_dataset = CustomDataset(config=data_loader_config, mode="train", cache=True)
        val_dataset = CustomDataset(config=data_loader_config, mode="val", cache=True)
        test_dataset = CustomDataset(config=data_loader_config, mode="test", cache=True)
        
        # Print summaries
        train_dataset.print_summary()
        val_dataset.print_summary()
        test_dataset.print_summary()

        # Initialize and train the model
        model = CustomModel(model_config)
        perform_cross_validation(model=model, train_dataset=train_dataset, val_dataset=val_dataset)
        
        # Define model directories for inference
        model_dirs = [f"{Paths.BEST_MODEL_CHECKPOINTS}/best_{model_config.MODEL}_{model_config.NAME}_{data_loader_config.NAME}_fold_{fold}.pth" for fold in range(model_config.FOLDS)]

        # Perform inference using the trained models
        perform_inference(test_dataset, model, model_dirs)

    except Exception as e:
        logger.error(f"Error training model {model_config.NAME} with data loader {data_loader_config.NAME}: {e}")


def grid_search(max_processes=4):
    # Determine the number of processes to use
    if max_processes is None:
        # Default to half the number of available CPUs to prevent overload
        max_processes = multiprocessing.cpu_count() // 2

    # Initialize the multiprocessing pool with a limited number of processes
    pool = multiprocessing.Pool(processes=max_processes)

    for data_loader_config in DATASET_GRID_SEARCH:
        process_function = partial(process_model_data_combo, data_loader_config)
        pool.map(process_function, MODEL_GRID_SEARCH)

    pool.close()
    pool.join()
