import multiprocessing

from datasets.data_loader import CustomDataset
from datasets.data_loader_configs import DATASET_SEARCH
from models.custom_model_configs import MODEL_SEARCH
from models.CustomModel import CustomModel
from utils.general_utils import get_logger
from utils.loader_utils import load_main_dfs
from utils.training_utils import train


def process_model_data_combo(
    data_loader_config, model_config, tensorboard_prefix="all"
):
    logger = get_logger("grid_search_process")

    try:
        logger.info(
            f"Training model {model_config.NAME} with data loader {data_loader_config.NAME}"
        )

        train_df, val_df, test_df = load_main_dfs(
            data_loader_config.SUBSET_SAMPLE_COUNT, train_val_split=(0.8, 0.2)
        )

        train_dataset = CustomDataset(
            config=data_loader_config,
            main_df=train_df,
            mode="train",
            cache=True,
            tensorboard_prefix=tensorboard_prefix,
        )
        val_dataset = CustomDataset(
            config=data_loader_config,
            main_df=val_df,
            mode="val",
            cache=True,
            tensorboard_prefix=tensorboard_prefix,
        )
        test_dataset = CustomDataset(
            config=data_loader_config,
            main_df=test_df,
            mode="test",
            cache=True,
            tensorboard_prefix=tensorboard_prefix,
        )

        train_dataset.print_summary()
        val_dataset.print_summary()
        test_dataset.print_summary()

        model = CustomModel(model_config, tensorboard_prefix=tensorboard_prefix)
        train(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tensorboard_prefix=tensorboard_prefix,
        )

    except Exception as e:
        logger.error(
            f"Error training model {model_config.NAME} with data loader {data_loader_config.NAME}: {e}"
        )


def grid_search(tensorboard_prefix="", max_processes=4):
    if max_processes is None:
        max_processes = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(processes=max_processes)

    for data_loader_config in DATASET_SEARCH:
        for model_config in MODEL_SEARCH:
            pool.apply_async(
                process_model_data_combo,
                args=(data_loader_config, model_config, tensorboard_prefix),
            )

    pool.close()
    pool.join()
