import pandas as pd
from logging import getLogger

def create_non_overlapping_eeg_crops(df: pd.DataFrame, label_cols: list) -> pd.DataFrame:
    """
    Preprocesses the EEG dataset to create a non-overlapping crop for each person.

    Args:
        df (pd.DataFrame): The input DataFrame containing EEG data.
        label_cols (list): List of column names representing labels in the DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with non-overlapping crops per person.

    Raises:
        ValueError: If the input DataFrame is empty or label_cols is not provided.
    """
    logger = getLogger('utils/create_non_overlapping_eeg_crops')
    if df.empty or not label_cols:
        logger.error("Input DataFrame is empty or label columns are not provided.")
        raise ValueError("Input DataFrame is empty or label columns are not provided.")

    try:
        crop_df = df.groupby("eeg_id")[["spectrogram_id", "spectrogram_label_offset_seconds"]] \
                    .agg({"spectrogram_id": "first", "spectrogram_label_offset_seconds": "min"})
        crop_df.columns = ["spectrogram_id", "min"]

        aux = df.groupby("eeg_id")[["spectrogram_id", "spectrogram_label_offset_seconds"]] \
                .agg({"spectrogram_label_offset_seconds": "max"})
        crop_df["max"] = aux["spectrogram_label_offset_seconds"]

        aux = df.groupby("eeg_id")[["patient_id"]].agg("first")
        crop_df["patient_id"] = aux["patient_id"]

        aux = df.groupby("eeg_id")[label_cols].agg("sum")
        for label in label_cols:
            crop_df[label] = aux[label].values

        y_data = crop_df[label_cols].values
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        crop_df[label_cols] = y_data

        aux = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
        crop_df["target"] = aux["expert_consensus"]

        crop_df.reset_index(inplace=True)
        logger.info(f"Train non-overlapping eeg_id shape: {crop_df.shape}")

        return crop_df
    except Exception as e:
        logger.error(f"Error in preprocessing data: {e}")
        raise
