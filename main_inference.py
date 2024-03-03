import pandas as pd
from datasets.data_loader import CustomDataset
from datasets.data_loader_configs import FullDataInferenceConfig, FullDataConfig
from generics import Paths, Generics
from models.CustomModel import CustomModel
from models.custom_model_configs import ShuffleNetBase
from utils.general_utils import get_logger
from utils.inference_utils import perform_inference, create_submission



def main(submission_file='submission.csv'):
    logger = get_logger("main")
    
    data_loader_config_test = FullDataInferenceConfig
    data_loader_config_train = FullDataConfig

    model_config = ShuffleNetBase
    
    logger.info(f"Performing inference on model {model_config.NAME} with data loader {data_loader_config_test.NAME}")
    
    test_df = pd.read_csv(Paths.TEST_CSV)
    
    # Load dataset
    test_dataset = CustomDataset(config=data_loader_config_test, main_df = test_df, mode="test", cache=True)
    
    # Print summaries
    test_dataset.print_summary()

    model = CustomModel(model_config)
    model_dir = f"{Paths.BEST_MODEL_CHECKPOINTS}best_{model_config.MODEL}_{model_config.NAME}_{data_loader_config_train.NAME}.pth"

    y_pred_probabilities = perform_inference(test_dataset, model, model_dir)
    
    submission_df = create_submission(test_dataset.main_df, y_pred_probabilities, Generics.LABEL_COLS, submission_file)
    print("Submission created!")

if __name__ == "__main__":
    main()