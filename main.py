import pandas as pd
from datasets.data_loader import CustomDataset
from generics import Paths
from models.CustomModel import CustomModel
from utils.general_utils import get_logger
from utils.training_utils import get_result, train_loop
from torch import nn
import torch
from datasets.data_loader_configs import ExtremelySmallBaseConfig
from models.custom_model_configs import EfficientNetB0Config

def main():
    logger = get_logger("main")
    data_loader_config = ExtremelySmallBaseConfig()
    train_dataset = CustomDataset(
        config=data_loader_config,mode="train", cache=True
    )
    train_dataset.print_summary()

    val_dataset = CustomDataset(
        config=data_loader_config, mode="val", cache=True
    )
    
    val_dataset.print_summary()
    
    
    test_dataset = CustomDataset(
        config=data_loader_config, mode="test", cache=True
    )
    
    test_dataset.print_summary()
    
    


    
    model_config = EfficientNetB0Config
    model = CustomModel(model_config)

    oof_df = pd.DataFrame()
    for fold in range(model_config.FOLDS):
        if fold in [0, 1, 2, 3, 4]:
            _oof_df = train_loop(
                train_dataset,
                val_dataset,
                model,
                fold,
            )
            oof_df = pd.concat([oof_df, _oof_df])
            logger.info(
                f"========== Fold {fold} result: {get_result(_oof_df)} =========="
            )
            print(f"========== Fold {fold} result: {get_result(_oof_df)} ==========")
    oof_df = oof_df.reset_index(drop=True)
    logger.info(f"========== CV: {get_result(oof_df)} ==========")
    oof_df.to_csv(Paths.OUTPUT_DIR + "/oof_df.csv", index=False)


if __name__ == "__main__":
    main()
