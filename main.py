
from datasets.data_loader import CustomDataset
from datasets.data_loader_configs import BaseDataConfig, Config_Normalize_Group_Raw_ICA, Config_Normalize_Group_Raw_MSPCA, SmallBaseConfig_Train_Annotator_Agreement, SmallConfig_Normalize_Group_Raw_ICA, ExtremelySmallBaseConfig
from generics import Paths
from models.CustomModel import CustomModel
from models.custom_model_configs import BaseModelConfig, EfficientNetB0Config_Big, EfficientNetB0Config_Big_Weight_Decay_FROZEN_32, EfficientNetB0Config_Big_Weight_Decay_Only_Custom_spectrograms
from utils.data_preprocessing_utils import filter_by_agreement, filter_by_annotators
from datasets.raw_data_loader import CustomRawDataset
from utils.general_utils import get_logger
from utils.grid_search_utils import grid_search
from utils.inference_utils import create_submission, perform_inference
from utils.loader_utils import load_main_dfs
from utils.training_utils import train
from generics import Generics

def main():
    class Config_Normalize_Group_Raw_ICA_50(BaseDataConfig):
        NORMALIZE_EEG_SPECTROGRAMS = True
        NORMALIZE_INDIVIDUALLY = False
        APPLY_ICA_RAW_EEG = True
        USE_PRELOADED_EEG_SPECTROGRAMS = False
        USE_PRELOADED_SPECTROGRAMS = False
        BATCH_SIZE_TEST = 1
        FILTER_BY_AGREEMENT = False
        FILTER_BY_AGREEMENT_MIN = 50
        BATCH_SIZE_TRAIN = 32
        BATCH_SIZE_VAL = 32
        FILTER_BY_AGREEMENT_ON_VAL = False
        SUBSET_SAMPLE_COUNT = 300


    class EfficientNetB0Config_Big_Weight_Decay(BaseModelConfig):
        MODEL = "tf_efficientnet_b0"
        FREEZE = False
        EPOCHS = 20
        GRADIENT_ACCUMULATION_STEPS = 1
        WEIGHT_DECAY = 0.01
        MAX_LEARNING_RATE_SCHEDULERER = 0.001
        USE_KAGGLE_SPECTROGRAMS = True

    data_loader_config = Config_Normalize_Group_Raw_ICA_50
    model_config = EfficientNetB0Config_Big_Weight_Decay
        
    train_df, val_df, test_df = load_main_dfs(data_loader_config, train_val_split=(0.8, 0.2))

    test_dataset = CustomDataset(config=data_loader_config,main_df = test_df, mode="test", cache=False)
    # val_dataset = CustomDataset(config=data_loader_config,main_df = val_df, mode="val", cache=False)
    # train_dataset = CustomDataset(config=data_loader_config,main_df = train_df, mode="train", cache=False)
    model = CustomModel(model_config, pretrained = False)
    # train(train_dataset=train_dataset, val_dataset=val_dataset, model=model,  tensorboard_prefix="50 samples") 
    modelDir = f"{Paths.BEST_MODEL_CHECKPOINTS}best_{model_config.MODEL}_{model_config.NAME}_{data_loader_config.NAME}.pth"

    preds = perform_inference(test_dataset, model, modelDir)

    create_submission(test_df, preds, Generics.LABEL_COLS, "submission.csv")

if __name__ == "__main__":
    main()