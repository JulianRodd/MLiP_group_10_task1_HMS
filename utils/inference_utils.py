import pandas as pd
import torch
from datasets.data_loader import CustomDataset
import os
from generics import Paths
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from torch import nn


def perform_inference(test_dataset: CustomDataset, model, model_dirs: list):
    """
    Perform inference on the test dataset using the trained model and log results to TensorBoard.

    Args:
        test_dataset (CustomDataset): The test dataset.
        model (torch.nn.Module): The base model on with which checkpoints were generated.
        model_dirs (list): list of paths to model checkpoints to test 

    Returns:
        Dictionary: A dictionary containing model predictions for the test set.
    """
    combined_writer = SummaryWriter(Paths.TENSORBOARD_INFERENCE, f"{model.config.NAME}_{test_dataset.config.NAME}_combined")
    combined_preds = []
    for model_dir in model_dirs:
        model.load_state_dict(torch.load(model_dir))
        model.eval()  # Set the model to evaluation mode
        test_loader = test_dataset.get_torch_data_loader()
        softmax = nn.Softmax(dim=1)
        prediction_dict = {}
        preds = []

        # Setup TensorBoard writer
        tb_run_path = os.path.join(
            Paths.TENSORBOARD_INFERENCE,
            f"{model.config.NAME}_{test_dataset.config.NAME}"
        )
        writer = SummaryWriter(tb_run_path)

        with tqdm(test_loader, unit="test_batch", desc='Inference') as tqdm_test_loader:
            for step, (X, _) in enumerate(tqdm_test_loader):
                X = X.to(model.device)
                with torch.no_grad():
                    y_preds = model(X)
                y_preds = softmax(y_preds)
                preds.append(y_preds.to('cpu').numpy())

                # Log predictions as histograms to TensorBoard
                for i, probability in enumerate(y_preds.to('cpu').numpy()):
                    writer.add_histogram(f"Predictions/Batch_{step}_Sample_{i}", probability, step)

        # Check if preds is empty
        if not preds:
            raise ValueError("No predictions were made by the model.")

        prediction_dict["predictions"] = np.concatenate(preds)
        writer.close()
        combined_preds.append(prediction_dict["predictions"])

    combined_preds = np.mean(combined_preds, axis=0)
    combined_writer.add_histogram("Predictions/Combined", combined_preds, 0)
    combined_writer.close()
    return combined_preds
  

def create_submission(test_df, predictions, target_columns, submission_file):
    """
    Creates a submission file from the predictions.

    Args:
        test_df (pd.DataFrame): The test DataFrame containing 'eeg_id'.
        predictions (np.ndarray): The predictions from the models.
        target_columns (list): List of target column names.
        submission_file (str): Path to the submission file.

    Returns:
        pd.DataFrame: The created submission DataFrame.
    """
    # Ensure the number of rows in predictions matches test_df
    assert len(test_df) == len(predictions), "Mismatch in number of predictions and number of test samples"

    # Create a DataFrame for submission
    submission_df = pd.DataFrame(predictions, columns=target_columns)
    submission_df['eeg_id'] = test_df['eeg_id'].values

    # Reorder the columns to have 'eeg_id' first
    column_order = ['eeg_id'] + target_columns
    submission_df = submission_df[column_order]

    # Save the submission file
    submission_df.to_csv(submission_file, index=False)

    print(f'Submission shape: {submission_df.shape}')
    return submission_df