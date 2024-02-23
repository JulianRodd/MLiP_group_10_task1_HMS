from torcheeg.transforms import BandHiguchiFractalDimension, BandPowerSpectralDensity
import numpy as np
import pandas as pd

def get_hfda(eeg, eeg_columns, k_max=6):
	transform = BandHiguchiFractalDimension()
	eeg_t = np.transpose(eeg)
	out = transform(eeg=eeg_t, K_max=k_max)["eeg"]
	df = pd.DataFrame(np.transpose(out), columns=eeg_columns)
	df.index = ["HDF_1", "HDF_2", "HDF_3", "HDF_4"]
	return df

def get_psd(eeg, eeg_columns):
	transform = BandPowerSpectralDensity()
	eeg_t = np.transpose(eeg)
	out = transform(eeg=eeg_t)["eeg"]
	df = pd.DataFrame(np.transpose(out), columns=eeg_columns)
	df.index = ["PSD_1", "PSD_2", "PSD_3", "PSD_4"]
	return df