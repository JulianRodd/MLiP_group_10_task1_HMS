import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from glob import glob
from generics import Generics
from mne.viz.utils import _compute_scalings
from mne._fiff.pick import channel_indices_by_type


main_df_path = "/home/janneke/Documents/Master/Machine_Learning_in_Practice/HMS/MLiP_group_10_task1_HMS/data/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
main_df = pd.read_csv(main_df_path)
num_subsamples = 10
subsample_paths = "/home/janneke/Documents/Master/Machine_Learning_in_Practice/HMS/MLiP_group_10_task1_HMS/data/cache/raw_subsamples/*.csv"
count = 0

for subsample_file in glob(subsample_paths):
    count += 1
    if count > num_subsamples:
        break
    eeg_id = int(subsample_file.split("/")[-1].split("_")[0])
    subsample_id = int(subsample_file.split("/")[-1].split("_")[-1].split(".")[0])
    
    print(f"{eeg_id}: {subsample_id}")
    subsample_info = main_df[main_df["eeg_id"]==eeg_id][main_df["eeg_sub_id"]==subsample_id]
    print(subsample_info[Generics.LABEL_COLS])
    
	# eeg = pd.read_parquet(raw_eeg_path)
    eeg = pd.read_csv(subsample_file, index_col=0)

    # Create MNE Info object
    sfreq = 200  # Sampling frequency in Hz
    ch_types = ["eeg"] * (len(eeg.columns) - 1)
    ch_types.append("ecg")
    info = mne.create_info(ch_names=eeg.columns.tolist(), sfreq=sfreq, ch_types=ch_types)

    # Convert pandas DataFrame to MNE RawArray
    raw: mne.io.RawArray = mne.io.RawArray(eeg.values.T, info)

    montage = mne.channels.make_standard_montage('standard_1020')

    raw = raw.set_montage(montage)
    updated_mont = raw.get_montage()

    # scale channels
    ch_types = channel_indices_by_type(raw.info)
    ch_types = {i_type: i_ixs for i_type, i_ixs in ch_types.items() if len(i_ixs) != 0}

    # for key, value in scalings.items():
    #     if key not in ch_types:
    #         continue
    #     values = raw[ch_types[key]]
    #     values = values.ravel()
    #     values = values[np.isfinite(values)]
    #     iqr = np.diff(np.percentile(values, [25, 75]))[0]
        

    # Plot EEG data in 10-20 format
    # raw.plot(scalings=dict(eeg=100, ecg=100), duration=50.0)  # Plot electrode locations with names
    # raw.plot_psd()
    # raw.plot_psd_topo()

    # use a bipolar reference (contralateral)
    # raw = mne.set_bipolar_reference(
    #     raw, 
    #     anode=  ["Fp1", "F3", "C3", "P3", "Fp2", "F4", "C4", "P4", "Fp1", "F7", "T3", "T5", "Fp2", "F8", "T4", "T6", "Fz", "Cz"], 
    #     cathode=["F3", "C3", "P3", "O1", "F4", "C4", "P4", "O2", "F7", "T3", "T5", "O1", "F8", "T4", "T6", "O2", "Cz", "Pz"])
    raw.plot_sensors(show_names=True)  # Plot electrode locations with names
    # raw_bip_ref.plot(scalings=dict(eeg=100, ecg=100), duration=50.0)
    # raw_bip_ref.plot(scalings=dict(eeg=100, ecg=100), duration=50.0)

    # print(mne.channels.get_builtin_montages())
    mne.channels.make_standard_montage("standard_1020")

    # plt.show()

    montage = raw.get_montage()
    print(montage.dig)
    plt.show()

    exit()
    # HFF and LFF
    raw.filter(l_freq=1, h_freq=70)

    # ica = mne.preprocessing.ICA(random_state=97, max_iter=800)
    # ica.fit(raw)
    # ica.exclude = [1, 2, 3]  #
    # raw.plot_sources(raw, show_scrollbars=False)
    # raw.plot_components()
    # orig_raw = raw.copy()
    # raw.load_data()

    raw.plot(scalings=dict(eeg=100, ecg=100), duration=50.0)
    raw.plot_psd()
    
    plt.show()
    plt.close()
