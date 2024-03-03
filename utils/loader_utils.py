import os
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from glob import glob
from utils.data_preprocessing_utils import filter_by_agreement, filter_by_annotators
from utils.eeg_processing_utils import generate_eeg_from_parquet, generate_spectrogram_from_eeg
from generics import Generics, Paths
from utils.general_utils import get_logger


import pandas as pd
import numpy as np
from typing import Dict
from glob import glob
from tqdm import tqdm
from utils.general_utils import get_logger
from generics import Paths
from utils.ica_utils import apply_ica_raw_eeg
from utils.mspca_utils import apply_mspca_raw_eeg


def load_main_dfs(data_loader_config, train_val_split = (0.8, 0.2)) -> pd.DataFrame:
    try:
        train_sample_count = data_loader_config.SUBSET_SAMPLE_COUNT
        
        logger = get_logger("main_df_loader.log")
        train_csv_pd = pd.read_csv(Paths.TRAIN_CSV)
        prepared_train_df = prepare_train_df(train_csv_pd)
        
        test_csv_pd = pd.read_csv(Paths.TEST_CSV)
        prepared_train_df= prepared_train_df[~prepared_train_df["eeg_id"].isin(Generics.OPT_OUT_EEG_ID)]
        test_csv_pd= test_csv_pd[~test_csv_pd["eeg_id"].isin(Generics.OPT_OUT_EEG_ID)]
        
        # Sample one record from each unique patient
        if data_loader_config.ONE_SAMPLE:
            sampled_train_csv_pd = prepared_train_df.groupby('patient_id').sample(n=1, random_state=42).reset_index(drop=True)
            samples_test_csv_pd = test_csv_pd.groupby('patient_id').sample(n=1, random_state=42).reset_index(drop=True)
        else: 
            sampled_train_csv_pd = prepared_train_df
            samples_test_csv_pd = test_csv_pd

        if train_sample_count == 0:
            train_sample_count = len(sampled_train_csv_pd)
        
        sampled_train_csv_pd = sampled_train_csv_pd.sample(n=train_sample_count, random_state=42)
            
        gss = GroupShuffleSplit(n_splits=2, train_size=train_val_split[0], random_state=42)
        splits = gss.split(sampled_train_csv_pd, groups=train_df.patient_id)

        train_id, val_id = next(splits)
        train_df = sampled_train_csv_pd.loc[train_id]
        val_df = sampled_train_csv_pd.loc[val_id]
        
        if (data_loader_config.FILTER_BY_AGREEMENT):
          train_df = filter_by_agreement(train_df, data_loader_config.FILTER_BY_AGREEMENT_MIN)
          if (data_loader_config.FILTER_BY_AGREEMENT_ON_VAL):
            val_df = filter_by_agreement(val_df, data_loader_config.FILTER_BY_AGREEMENT_MIN)
        
        if (data_loader_config.FILTER_BY_ANNOTATOR):
          train_df = filter_by_annotators(train_df, data_loader_config.FILTER_BY_ANNOTATOR_MIN, data_loader_config.FILTER_BY_ANNOTATOR_MAX, n_annot=train_df['n_annot'])
          if (data_loader_config.FILTER_BY_ANNOTATOR_ON_VAL):
            val_df = filter_by_annotators(val_df, data_loader_config.FILTER_BY_ANNOTATOR_MIN, data_loader_config.FILTER_BY_ANNOTATOR_MAX, n_annot=val_df['n_annot'])
        test_df = samples_test_csv_pd

        return train_df, val_df, test_df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
      
def prepare_train_df(df: pd.DataFrame) -> pd.DataFrame:
    train_df = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg({
    'spectrogram_id':'first',
    'spectrogram_label_offset_seconds':'min'
    })
    train_df.columns = ['spectrogram_id','min']

    aux = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg({
        'spectrogram_label_offset_seconds':'max'
    })
    train_df['max'] = aux

    aux = df.groupby('eeg_id')[['patient_id']].agg('first')
    train_df['patient_id'] = aux

    aux = df.groupby('eeg_id')[Generics.LABEL_COLS].agg('sum')
    for label in Generics.LABEL_COLS:
        train_df[label] = aux[label].values
        
    y_data = train_df[Generics.LABEL_COLS].values
    train_df['n_annot'] = y_data.sum(axis=1)
    y_data = y_data / y_data.sum(axis=1,keepdims=True)
    train_df[Generics.LABEL_COLS] = y_data
    
    aux = df.groupby('eeg_id')[['expert_consensus']].agg('first')
    train_df['target'] = aux

    train_df = train_df.reset_index()
    print('Train non-overlapp eeg_id shape:', train_df.shape )
    train_df.head()
    return train_df
    
    
def load_eeg_data(main_df: pd.DataFrame, mode: str) -> Dict[int, pd.DataFrame]:
    """
    Load EEG data for the EEG IDs present in the provided DataFrame.

    Args:
        main_df (pd.DataFrame): DataFrame containing the 'eeg_id' column.
        mode (str): Operating mode ('train' or 'test').

    Returns:
        Dict[int, pd.DataFrame]: Dictionary of EEG data keyed by EEG ID.
    """
    logger = get_logger("eeg_data_loader.log")
    try:
        eeg_ids = set(main_df["eeg_id"])
        csv_path = Paths.TEST_EEGS if mode == "test" else Paths.TRAIN_EEGS
        paths_eegs = [f for f in glob(f"{csv_path}*.parquet") if int(os.path.basename(f).split(".")[0]) in eeg_ids]

        logger.info(f"Loading {len(paths_eegs)} EEGs out of {len(eeg_ids)} available in dataset")
        eeg_data_dict = {}

        for file_path in tqdm(paths_eegs):
            eeg_id = int(os.path.basename(file_path).split(".")[0])
            eeg_data = generate_eeg_from_parquet(file_path)
            eeg_data_dict[eeg_id] = eeg_data

        return eeg_data_dict

    except Exception as e:
        logger.error(f"Error loading EEG data: {e}")
        raise

def process_eeg_data(eeg_data: pd.DataFrame, feats: list, use_wavelet: bool, mspca_on_raw_eeg: bool, ica_on_raw_eeg: bool) -> np.ndarray:
    """
    Process EEG data to generate a spectrogram, and apply MSPCA or ICA if required.

    Args:
        eeg_data (pd.DataFrame): Raw EEG data.
        feats: List of features or electrodes used for EEG.
        use_wavelet (bool): Whether to use wavelet denoising.
        mspca_on_raw_eeg (bool): Apply MSPCA on raw EEG data.
        ica_on_raw_eeg (bool): Apply ICA on raw EEG data.

    Returns:
        np.ndarray: Processed EEG data.
    """
    if mspca_on_raw_eeg:
        eeg_data = apply_mspca_raw_eeg(eeg_data)
    elif ica_on_raw_eeg:
        eeg_data = apply_ica_raw_eeg(eeg_data, n_components=4)
        
    # Generate EEG spectrogram
    eeg_spectrogram = generate_spectrogram_from_eeg(eeg_data, feats, use_wavelet)

    return eeg_spectrogram


def load_eeg_spectrograms(main_df: pd.DataFrame, mode: str, feats, use_wavelet, mspca_on_raw_eeg: bool, ica_on_raw_eeg: bool) -> Dict[int, np.ndarray]:
    """
    Load EEG spectrograms for the EEG IDs present in the provided DataFrame.

    Args:
        main_df (pd.DataFrame): DataFrame containing the 'eeg_id' column.
        mode (str): Operating mode ('train' or 'test').
        feats: List of features or electrodes used for EEG.
        use_wavelet (bool): Whether to use wavelet denoising.
        mspca_on_raw_eeg (bool): Apply MSPCA on raw EEG data.
        ica_on_raw_eeg (bool): Apply ICA on raw EEG data.

    Returns:
        Dict[int, np.ndarray]: Dictionary of EEG spectrograms keyed by EEG ID.
    """
    logger = get_logger("eeg_spectrogram_loader.log")
    try:
        # Load raw EEG data
        eeg_data_dict = load_eeg_data(main_df, mode)

        # Process each EEG data
        eeg_spectrograms = {}
        for eeg_id, eeg_data in tqdm(eeg_data_dict.items(), desc="Processing EEG Data"):
            eeg_id = int(eeg_id)
            eeg_spectrogram = process_eeg_data(eeg_data, feats, use_wavelet, mspca_on_raw_eeg, ica_on_raw_eeg)
            eeg_spectrograms[eeg_id] = eeg_spectrogram

        return eeg_spectrograms

    except Exception as e:
        logger.error(f"Error loading EEG spectrograms: {e}, {e.args}")
        raise

def load_preloaded_eeg_spectrograms(main_df: pd.DataFrame):
    pre_loaded_eegs = np.load(Paths.PRE_LOADED_EEGS, allow_pickle=True).item()
    # select only where in main_df
    return {k: v for k, v in pre_loaded_eegs.items() if k in main_df["eeg_id"].values}

def normalize_eeg_spectrograms(eeg_spectrograms: Dict[int, np.ndarray], normalize_indiv = False) -> Dict[int, np.ndarray]:
    """
    Normalize EEG data in a dictionary by subtracting the mean and dividing by the standard deviation.

    Parameters:
    all_eegs (dict): A dictionary containing EEG data with shape (128, 256, 4) for each entry.

    Returns:
    dict: A new dictionary containing normalized EEG data.
    """
    normalized_eegs = {}
    
    if normalize_indiv:
      for eeg_id, eeg_data in eeg_spectrograms.items():
          mean = np.mean(eeg_data, axis=(0, 1))
          std = np.std(eeg_data, axis=(0, 1))
          normalized_eeg = (eeg_data - mean) / std
          normalized_eegs[eeg_id] = normalized_eeg
    else:
      eeg_data = np.array([eeg_spectrograms[eeg_id] for eeg_id in eeg_spectrograms.keys()])
      mean = np.mean(eeg_data, axis=(0, 1, 2))
      std = np.std(eeg_data, axis=(0, 1, 2))
      for eeg_id, eeg_data in eeg_spectrograms.items():
          normalized_eeg = (eeg_data - mean) / std
          normalized_eegs[eeg_id] = normalized_eeg
      
    return normalized_eegs

def load_preloaded_spectrograms(main_df: pd.DataFrame):
    pre_loaded_spectrograms = np.load(Paths.PRE_LOADED_SPECTROGRAMS, allow_pickle=True).item()
    # select only where in main_df
    return {k: v for k, v in pre_loaded_spectrograms.items() if k in main_df["spectrogram_id"].values}
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
