import numpy as np
from sklearn.decomposition import PCA
from pywt import wavedec
from utils.general_utils import get_logger

# Initialize logger
logger = get_logger("utils/mspca_utils")

def apply_mspca_to_eeg_spectrograms(eeg_spectrograms, n_components=10, wavelet='db4', level=5):
    """
    Apply Multiscale Principal Component Analysis (MSPCA) to EEG spectrogram data.

    Args:
        eeg_spectrograms (dict): Dictionary containing EEG spectrogram data.
        n_components (int): Number of principal components to keep at each scale.
        wavelet (str): Type of wavelet to use for multiscale decomposition.
        level (int): Decomposition level.

    Returns:
        dict: Dictionary containing transformed EEG spectrograms after applying MSPCA.
    """
    mspca_processed_eegs = {}

    for eeg_id, eeg_data in eeg_spectrograms.items():
        logger.info(f"Processing EEG {eeg_id}")

        try:
            # Decompose EEG spectrogram using wavelet transform
            coeffs = wavedec(eeg_data, wavelet=wavelet, level=level, axis=-1)
            transformed_coeffs = []

            # Apply PCA on each scale
            for coeff in coeffs:
                pca = PCA(n_components=n_components)
                transformed_coeff = pca.fit_transform(coeff)
                transformed_coeffs.append(transformed_coeff)

            # Concatenate the transformed coefficients from each scale
            reconstructed_eeg = np.concatenate(transformed_coeffs, axis=-1)

            mspca_processed_eegs[eeg_id] = reconstructed_eeg
        except ValueError as e:
            logger.error(f"Failed to process EEG {eeg_id}: {e}")

    logger.info("Finished processing all EEGs.")
    return mspca_processed_eegs

