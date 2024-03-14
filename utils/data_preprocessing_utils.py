import numpy as np
import pandas as pd

from generics import Generics
from utils.general_utils import get_logger


def create_non_overlapping_eeg_crops(
    df: pd.DataFrame, label_cols: list
) -> pd.DataFrame:
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
    logger = get_logger("data_preprocessing/create_non_overlapping_eeg_crops")
    if df.empty or not label_cols:
        logger.error("Input DataFrame is empty or label columns are not provided.")
        raise ValueError("Input DataFrame is empty or label columns are not provided.")

    try:
        crop_df = df.groupby("eeg_id")[
            ["spectrogram_id", "spectrogram_label_offset_seconds"]
        ].agg({"spectrogram_id": "first", "spectrogram_label_offset_seconds": "min"})
        crop_df.columns = ["spectrogram_id", "min"]

        aux = df.groupby("eeg_id")[
            ["spectrogram_id", "spectrogram_label_offset_seconds"]
        ].agg({"spectrogram_label_offset_seconds": "max"})
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


def filter_by_agreement(df: pd.DataFrame, min: float):
    """
    Takes train/test df
    Returns df with rows having more than min agreement (min in %).
    """

    max_votes = df[Generics.LABEL_COLS].max(axis=1)
    total_votes = df[Generics.LABEL_COLS].sum(axis=1)
    min /= 100
    bool_filter = (max_votes / total_votes) > min

    return df[bool_filter]


def filter_by_annotators(df: pd.DataFrame, min: int, max: int = np.inf, n_annot=None):
    """
    Takes train/test df, min and max number of annotators
    Returns df with rows having more than or min number of annotators and less than max
    based on vote columns or n_annot, if provided
    """

    if n_annot is not None:
        total_votes = n_annot
    else:
        total_votes = df[Generics.LABEL_COLS].sum(axis=1)
    filter_min = total_votes >= min
    filter_max = total_votes < max

    bool_filter = filter_min & filter_max

    return df[bool_filter]
