import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from utils.signal_preprocessing_utils import denoise
from utils.general_utils import get_logger

import pandas as pd
import numpy as np
import librosa
from utils.general_utils import get_logger

def generate_eeg_from_parquet(parquet_path: str) -> pd.DataFrame:
    """
    Reads EEG data from a Parquet file.

    Args:
        parquet_path (str): Path to the Parquet file containing EEG data.

    Returns:
        pd.DataFrame: DataFrame containing EEG data.

    Raises:
        FileNotFoundError: If the specified Parquet file does not exist.
        Exception: For issues encountered during reading the Parquet file.
    """
    logger = get_logger("utils/generate_eeg_from_parquet")
    try:
        eeg = pd.read_parquet(parquet_path)
        return eeg
    except FileNotFoundError:
        logger.error(f"Parquet file not found: {parquet_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading Parquet file: {e}")
        raise

def generate_spectrogram_from_eeg(eeg_data, feats, use_wavelet: bool, display: bool = False) -> np.ndarray:
    logger = get_logger("utils/generate_spectrogram_from_eeg")
    try:
        # Validate feats list
        if not all(len(sublist) == 5 for sublist in feats):
            raise ValueError("Each sublist in feats must contain 5 column names")

        # Validate columns in eeg_data
        for sublist in feats:
            for col in sublist:
                if col not in eeg_data.columns:
                    raise ValueError(f"Column '{col}' not found in eeg_data")
                  
        middle = (len(eeg_data) - 10_000) // 2
        eeg = eeg_data.iloc[middle : middle + 10_000]

        img = np.zeros((128, 256, 4), dtype="float32")
        signals = []

        for k in range(4):
            COLS = feats[k]

            for kk in range(4):
                x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values
                m = np.nanmean(x)
                x = np.nan_to_num(x, nan=m)

                if use_wavelet:
                    x = denoise(
                        x, wavelet=use_wavelet
                    )  # denoise function should be imported or defined

                signals.append(x)

                mel_spec = librosa.feature.melspectrogram(
                    y=x,
                    sr=200,
                    hop_length=len(x) // 256,
                    n_fft=1024,
                    n_mels=128,
                    fmin=0,
                    fmax=20,
                    win_length=128,
                )
                width = (mel_spec.shape[1] // 32) * 32
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(
                    np.float32
                )[:, :width]
                mel_spec_db = (mel_spec_db + 40) / 40
                img[:, :, k] += mel_spec_db

            img[:, :, k] /= 4.0

        if display:
            _display_eeg_and_spectrogram(img, signals, eeg_data["eeg_id"].values[0])

        return img
    except Exception as e:
        logger.error(f"Error processing EEG data: {e}")
        raise



def _display_eeg_and_spectrogram(
    img: np.ndarray, signals: list, display_eeg_id: int, config
):
    """
    Helper function to display EEG signals and spectrograms.

    Args:
        img (np.ndarray): Spectrogram image.
        signals (list): List of EEG signals.
        display_eeg_id (int): EEG ID to display in plot titles.
    """
    plt.figure(figsize=(10, 7))
    for k in range(4):
        plt.subplot(2, 2, k + 1)
        plt.imshow(img[:, :, k], aspect="auto", origin="lower")
        plt.title(f"EEG {display_eeg_id} - Spectrogram {config.NAMES[k]}")
    plt.show()

    plt.figure(figsize=(10, 5))
    offset = 0
    for k in range(4):
        if k > 0:
            offset -= signals[3 - k].min()
        plt.plot(
            range(10_000),
            signals[k] + offset,
            label=config.NAMES[3 - k],
        )
        offset += signals[3 - k].max()
    plt.legend()
    plt.title(f"EEG {display_eeg_id} Signals")
    plt.show()
