import gc

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

from utils.general_utils import get_logger

logger = get_logger("utils/ica_utils")


def apply_ica_to_eeg_spectrograms(
    eeg_spectrograms, n_components=4, max_iter=1000, tol=0.001
):
    ica_processed_eegs = {}

    for eeg_id, eeg_data in eeg_spectrograms.items():
        logger.info(f"Processing EEG {eeg_id}")

        eeg_data = np.nan_to_num(eeg_data)

        reshaped_eeg_data = eeg_data.reshape(-1, eeg_data.shape[2])

        ica = FastICA(
            n_components=n_components,
            whiten="unit-variance",
            max_iter=max_iter,
            tol=tol,
        )
        try:
            S_ = ica.fit_transform(reshaped_eeg_data)

            reconstructed_eeg = S_.reshape(eeg_data.shape)

            ica_processed_eegs[eeg_id] = reconstructed_eeg
        except ValueError as e:
            logger.log(f"Failed to converge for EEG {eeg_id}: {e}")

        del reshaped_eeg_data, S_, reconstructed_eeg
        gc.collect()

    logger.info("Finished processing all EEGs.")
    return ica_processed_eegs


def apply_ica_raw_eeg(eeg_data: pd.DataFrame, n_components=10) -> pd.DataFrame:
    try:

        imputer = SimpleImputer(strategy="mean")
        eeg_data_imputed = imputer.fit_transform(eeg_data)

        sel = VarianceThreshold(threshold=0)
        eeg_data_non_constant = sel.fit_transform(eeg_data_imputed)
        non_constant_cols = eeg_data.columns[sel.get_support()]

        if n_components > len(non_constant_cols):
            raise ValueError(
                "Number of components for ICA cannot be greater than the number of non-constant features in eeg_data"
            )

        ica = FastICA(n_components=n_components, random_state=0)
        S_ = ica.fit_transform(eeg_data_non_constant)

        reconstructed_eeg = ica.inverse_transform(S_)

        reconstructed_df = pd.DataFrame(
            reconstructed_eeg, columns=non_constant_cols, index=eeg_data.index
        )

        for col in eeg_data.columns:
            if col not in reconstructed_df.columns:
                reconstructed_df[col] = np.nan

        del eeg_data_imputed, eeg_data_non_constant, S_, reconstructed_eeg
        gc.collect()

        return reconstructed_df
    except Exception as e:
        print(f"An error occurred during ICA: {e}")
        return eeg_data
