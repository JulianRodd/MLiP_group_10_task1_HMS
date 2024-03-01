
import numpy as np
from sklearn.decomposition import FastICA
from pandas import DataFrame
from utils.general_utils import get_logger
from sklearn.decomposition import FastICA
from sklearn.impute import SimpleImputer
import pandas as pd
# Initialize logger
logger = get_logger("utils/ica_utils")

def apply_ica_to_eeg_spectrograms(eeg_spectrograms, n_components=4, max_iter=1000, tol=0.001):
    """
    Applies Independent Component Analysis (ICA) to EEG spectrogram data.
    """
    ica_processed_eegs = {}

    for eeg_id, eeg_data in eeg_spectrograms.items():
        logger.info(f"Processing EEG {eeg_id}")

        # Reshape eeg_data for ICA
        reshaped_eeg_data = eeg_data.reshape(-1, eeg_data.shape[2])

        # Perform ICA
        ica = FastICA(n_components=n_components, max_iter=max_iter, tol=tol)
        try:
            S_ = ica.fit_transform(reshaped_eeg_data)  # Get independent components

            # Reshape independent components back to original shape
            reconstructed_eeg = S_.reshape(eeg_data.shape)

            ica_processed_eegs[eeg_id] = reconstructed_eeg
        except ValueError as e:
            logger.error(f"Failed to converge for EEG {eeg_id}: {e}")

    logger.info("Finished processing all EEGs.")
    return ica_processed_eegs
  
  

from sklearn.feature_selection import VarianceThreshold


def apply_ica_raw_eeg(eeg_data: pd.DataFrame, n_components=10) -> pd.DataFrame:
    """
    Apply Independent Component Analysis (ICA) to EEG data after handling NaN values.

    Args:
        eeg_data (pd.DataFrame): Raw EEG data.
        n_components (int): Number of components to keep for ICA.

    Returns:
        pd.DataFrame: Processed EEG data after ICA, with the same shape as input.
    """
    # Impute NaN values
    imputer = SimpleImputer(strategy='mean')
    eeg_data_imputed = imputer.fit_transform(eeg_data)

    # Remove constant features but preserve columns
    sel = VarianceThreshold(threshold=0)
    eeg_data_non_constant = sel.fit_transform(eeg_data_imputed)
    non_constant_cols = eeg_data.columns[sel.get_support()]

    # Check if n_components is not greater than the number of features in eeg_data
    if n_components > len(non_constant_cols):
        raise ValueError("Number of components for ICA cannot be greater than the number of non-constant features in eeg_data")

    # Perform ICA
    ica = FastICA(n_components=n_components, random_state=0)
    S_ = ica.fit_transform(eeg_data_non_constant)  # Get the independent components

    # Transform back to the original space
    reconstructed_eeg = ica.inverse_transform(S_)

    # Create a DataFrame with the original column names
    reconstructed_df = pd.DataFrame(reconstructed_eeg, columns=non_constant_cols, index=eeg_data.index)
    
    # Re-add any dropped columns as NaNs to maintain the original DataFrame structure
    for col in eeg_data.columns:
        if col not in reconstructed_df.columns:
            reconstructed_df[col] = np.nan

    return reconstructed_df