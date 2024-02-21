
from sklearn.decomposition import FastICA

from utils.general_utils import get_logger

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