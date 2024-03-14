import multiprocessing
from functools import partial
import pandas as pd
from datasets.data_loader import CustomDataset
from generics import Paths
from models.CustomModel import CustomModel
from utils.general_utils import get_logger
from utils.inference_utils import perform_inference
from utils.loader_utils import load_main_dfs
from utils.training_utils import train
from datasets.data_loader_configs import DATASET_GRID_SEARCH
from models.custom_model_configs import MODEL_GRID_SEARCH

def process_model_data_combo(data_loader_config, model_config, tensorboard_prefix="all"):
    # Initialize the logger within each process
    logger = get_logger("grid_search_process")

    try:
        logger.info(f"Training model {model_config.NAME} with data loader {data_loader_config.NAME}")
        
        train_df, val_df, test_df = load_main_dfs(data_loader_config.SUBSET_SAMPLE_COUNT, train_val_split=(0.8, 0.2))
        # Load datasets
        train_dataset = CustomDataset(config=data_loader_config, main_df = train_df, mode="train", cache=True, tensorboard_prefix=tensorboard_prefix)
        val_dataset = CustomDataset(config=data_loader_config,main_df = val_df, mode="val", cache=True, tensorboard_prefix=tensorboard_prefix)
        test_dataset = CustomDataset(config=data_loader_config,main_df = test_df, mode="test", cache=True, tensorboard_prefix=tensorboard_prefix)
        
        # Print summaries
        train_dataset.print_summary()
        val_dataset.print_summary()
        test_dataset.print_summary()

        # Initialize and train the model
        model = CustomModel(model_config, tensorboard_prefix=tensorboard_prefix)
        train(model=model, train_dataset=train_dataset, val_dataset=val_dataset, tensorboard_prefix=tensorboard_prefix)

    except Exception as e:
        logger.error(f"Error training model {model_config.NAME} with data loader {data_loader_config.NAME}: {e}")


def grid_search(tensorboard_prefix = "", max_processes=4):
    # Determine the number of processes to use
    if max_processes is None:
        # Default to half the number of available CPUs to prevent overload
        max_processes = multiprocessing.cpu_count() // 2

    # Initialize the multiprocessing pool with a limited number of processes
    pool = multiprocessing.Pool(processes=max_processes)

    # Iterate over dataset configurations first, then model configurations
    for data_loader_config in DATASET_GRID_SEARCH:
        for model_config in MODEL_GRID_SEARCH:
            # Call the function with both configurations
            pool.apply_async(process_model_data_combo, args=(data_loader_config, model_config, tensorboard_prefix))

    pool.close()
    pool.join()
