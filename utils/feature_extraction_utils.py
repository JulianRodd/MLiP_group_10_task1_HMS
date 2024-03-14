import numpy as np
import pandas as pd
from torcheeg.transforms import (
    BandDetrendedFluctuationAnalysis,
    BandHiguchiFractalDimension,
    BandPowerSpectralDensity,
)


def get_hfda(eeg, eeg_columns, k_max=6):
    """Calculates the higuchi fractal dimension (HFD) of EEG signals in several sub-bands."""
    transform = BandHiguchiFractalDimension(sampling_rate=200)
    eeg_t = np.transpose(eeg)
    out = transform(eeg=eeg_t, K_max=k_max)["eeg"]
    df = pd.DataFrame(np.transpose(out), columns=eeg_columns)
    df.index = ["HDF_1", "HDF_2", "HDF_3", "HDF_4"]
    return df


def get_psd(eeg: np.ndarray, eeg_columns: list) -> pd.DataFrame:
    """Calculates the power spectral density of several sub-bands for all electrodes."""
    transform = BandPowerSpectralDensity(sampling_rate=200)
    eeg_t = np.transpose(eeg)
    out = transform(eeg=eeg_t)["eeg"]
    df = pd.DataFrame(np.transpose(out), columns=eeg_columns)
    df.index = ["PSD_1", "PSD_2", "PSD_3", "PSD_4"]
    return df


def get_dfa(eeg: np.ndarray, eeg_columns: list) -> pd.DataFrame:
    """Calculates the detrended fluctuation analysis (DFA) of EEG signals in several sub-bands."""
    transform = BandDetrendedFluctuationAnalysis(sampling_rate=200)
    eeg_t = np.transpose(eeg)
    out = transform(eeg=eeg_t)["eeg"]
    df = pd.DataFrame(np.transpose(out), columns=eeg_columns)
    df.index = ["DFA_1", "DFA_2", "DFA_3", "DFA_4"]
    return df
