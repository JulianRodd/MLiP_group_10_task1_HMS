from typing import Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from utils.eeg_processing_utils import generate_spectrogram_from_eeg
from generics import Paths
from utils.general_utils import get_logger


def load_eeg_spectrograms(main_df: pd.DataFrame, mode: str, feats, use_wavelet) -> Dict[int, np.ndarray]:
    """
    Load EEG spectrograms for the EEG IDs present in the provided DataFrame.

    Args:
        main_df (pd.DataFrame): DataFrame containing the 'eeg_id' column.
        mode (str): Operating mode ('train' or 'test').

    Returns:
        Dict[int, np.ndarray]: Dictionary of EEG spectrograms keyed by EEG ID.
    """
    logger = get_logger("eeg_spectrogram_loader.log")
    try:
        eeg_ids = set(main_df["eeg_id"])
        csv_path = Paths.TEST_EEGS if mode == "test" else Paths.TRAIN_EEGS
        paths_eegs = [
            f
            for f in glob(csv_path + "*.parquet")
            if int(f.split("/")[-1].split(".")[0]) in eeg_ids
        ]

        logger.info(
            f"Loading {len(paths_eegs)} EEGs out of {len(eeg_ids)} available in dataset"
        )
        eeg_spectrograms = {}

        for file_path in tqdm(paths_eegs):
            eeg_id = int(file_path.split("/")[-1].split(".")[0])
            eeg_spectrogram = generate_spectrogram_from_eeg(file_path, feats, use_wavelet)
            eeg_spectrograms[eeg_id] = eeg_spectrogram

        return eeg_spectrograms

    except Exception as e:
        logger.error(f"Error loading eeg_spectrograms: {e}")
        raise


def load_spectrograms(main_df: pd.DataFrame, mode: str) -> Dict[int, np.ndarray]:
    """
    Load spectrogram data for the spectrogram IDs present in the provided DataFrame.

    Args:
        main_df (pd.DataFrame): DataFrame containing the 'spectrogram_id' column.
        mode (str): Operating mode ('train' or 'test').

    Returns:
        Dict[int, np.ndarray]: Dictionary of spectrograms keyed by spectrogram ID.
    """
    logger = get_logger('spectrogram_loader.log')
    try:
        spectrogram_ids = set(main_df["spectrogram_id"])
        paths_spectrograms = [
            f for f in glob(Paths.TEST_SPECTROGRAMS + "*.parquet" if mode == "test" else Paths.TRAIN_SPECTROGRAMS + "*.parquet")
            if int(f.split("/")[-1].split(".")[0]) in spectrogram_ids
        ]
        
        logger.info(f"Loading {len(paths_spectrograms)} spectrograms out of {len(spectrogram_ids)} available in dataset")
        all_spectrograms = {}

        for file_path in tqdm(paths_spectrograms, desc="Loading Spectrograms"):
            spectrogram_id = int(file_path.split("/")[-1].split(".")[0])
            aux = pd.read_parquet(file_path)
            all_spectrograms[spectrogram_id] = aux.iloc[:, 1:].values
            del aux

        return all_spectrograms

    except FileNotFoundError as e:
        logger.error(f"Spectrogram loading error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading spectrograms: {e}")
        raise
