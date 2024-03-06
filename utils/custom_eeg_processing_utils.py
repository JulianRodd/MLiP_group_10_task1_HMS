import mne
import numpy as np
import pandas as pd
import librosa


def generate_custom_spectrogram_from_eeg(eeg_data, feats, custom_config):
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
        
        ################################################################################################################
        
        validate_config(custom_config)

        ch_types = ["eeg"] * (len(eeg.columns) - 1)
        ch_types.append("ecg")
        
        with mne.use_log_level("warning"):
            info = mne.create_info(ch_names=eeg.columns.tolist(), sfreq=custom_config["sfreq"], ch_types=ch_types)
        
            raw: mne.io.RawArray = mne.io.RawArray(eeg.values.T, info)
            
            montage = mne.channels.make_standard_montage('standard_1020')
            raw = raw.set_montage(montage)
            
            l_freq = custom_config.get("l_freq")
            h_freq = custom_config.get("h_freq")
            if l_freq is not None or h_freq is not None:
                raw = raw.filter(l_freq=l_freq, h_freq=h_freq)
                
            notch_filter = custom_config.get("notch_filter")
            if notch_filter is not None:
                raw.notch_filter(notch_filter)
                
                
            raw_np = raw.get_data()
            eeg_np = np.transpose(raw_np)
        eeg = pd.DataFrame(eeg_np, columns=eeg.columns)

        ################################################################################################################

        img = np.zeros((128, 256, 4), dtype="float32")
        signals = []

        for k in range(4):
            COLS = feats[k]

            for kk in range(4):
                x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values
                m = np.nanmean(x)
                x = np.nan_to_num(x, nan=m)

                # if use_wavelet:
                #     x = denoise(
                #         x, wavelet=use_wavelet
                #     )  # denoise function should be imported or defined

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

        # if display:
        #     _display_eeg_and_spectrogram(img, signals, eeg_data["eeg_id"].values[0])

        return img
    except Exception as e:
        print(f"Error processing EEG data: {e}")
        raise




def validate_config(config):
    if "sfreq" not in config:
        raise ValueError(f"eeg sample frequency ('sfreq') not in custom_preprocessing_config.")
    return












# main_df_path = "/home/janneke/Documents/Master/Machine_Learning_in_Practice/HMS/MLiP_group_10_task1_HMS/data/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
# main_df = pd.read_csv(main_df_path)
# num_subsamples = 10
# subsample_paths = "/home/janneke/Documents/Master/Machine_Learning_in_Practice/HMS/MLiP_group_10_task1_HMS/data/cache/raw_subsamples/*.csv"
# count = 0

# chris_preloaded = np.load(
#     "/home/janneke/Documents/Master/Machine_Learning_in_Practice/HMS/MLiP_group_10_task1_HMS/data/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy",
#     allow_pickle=True
#     ).item()


# for subsample_file in glob(subsample_paths):
#     count += 1
#     if count > num_subsamples:
#         break
#     eeg_id = int(subsample_file.split("/")[-1].split("_")[0])
#     subsample_id = int(subsample_file.split("/")[-1].split("_")[-1].split(".")[0])
    
#     print(f"{eeg_id}: {subsample_id}")
#     subsample_info = main_df[main_df["eeg_id"]==eeg_id][main_df["eeg_sub_id"]==subsample_id]
#     print(subsample_info[Generics.LABEL_COLS])
    
#     # eeg = pd.read_parquet(raw_eeg_path)
#     eeg = pd.read_csv(subsample_file, index_col=0)

#     # Create MNE Info object
#     sfreq = 200  # Sampling frequency in Hz
#     ch_types = ["eeg"] * (len(eeg.columns) - 1)
#     ch_types.append("ecg")
#     info = mne.create_info(ch_names=eeg.columns.tolist(), sfreq=sfreq, ch_types=ch_types)

#     # Convert pandas DataFrame to MNE RawArray
#     raw: mne.io.RawArray = mne.io.RawArray(eeg.values.T, info)

#     montage = mne.channels.make_standard_montage('standard_1020')

#     raw = raw.set_montage(montage)
#     updated_mont = raw.get_montage()

#     # scale channels
#     ch_types = channel_indices_by_type(raw.info)
#     ch_types = {i_type: i_ixs for i_type, i_ixs in ch_types.items() if len(i_ixs) != 0}

#     # for key, value in scalings.items():
#     #     if key not in ch_types:
#     #         continue
#     #     values = raw[ch_types[key]]
#     #     values = values.ravel()
#     #     values = values[np.isfinite(values)]
#     #     iqr = np.diff(np.percentile(values, [25, 75]))[0]
        

#     # Plot EEG data in 10-20 format
#     # raw.plot(scalings=dict(eeg=100, ecg=100), duration=50.0)  # Plot electrode locations with names
#     # raw.plot_psd()
#     # raw.plot_psd_topo()

#     # use a bipolar reference (contralateral)
#     # raw = mne.set_bipolar_reference(
#     #     raw, 
#     #     anode=  ["Fp1", "F3", "C3", "P3", "Fp2", "F4", "C4", "P4", "Fp1", "F7", "T3", "T5", "Fp2", "F8", "T4", "T6", "Fz", "Cz"], 
#     #     cathode=["F3", "C3", "P3", "O1", "F4", "C4", "P4", "O2", "F7", "T3", "T5", "O1", "F8", "T4", "T6", "O2", "Cz", "Pz"])
#     raw.plot_sensors(show_names=True)  # Plot electrode locations with names
#     # raw_bip_ref.plot(scalings=dict(eeg=100, ecg=100), duration=50.0)
#     # raw_bip_ref.plot(scalings=dict(eeg=100, ecg=100), duration=50.0)

#     # print(mne.channels.get_builtin_montages())
#     mne.channels.make_standard_montage("standard_1020")

#     # plt.show()

#     montage = raw.get_montage()
#     print(montage.dig)
#     plt.show()

#     # HFF and LFF
#     raw.filter(l_freq=1, h_freq=70)

#     # ica = mne.preprocessing.ICA(random_state=97, max_iter=800)
#     # ica.fit(raw)
#     # ica.exclude = [1, 2, 3]  #
#     # raw.plot_sources(raw, show_scrollbars=False)
#     # raw.plot_components()
#     # orig_raw = raw.copy()
#     # raw.load_data()

#     raw.plot(scalings=dict(eeg=100, ecg=100), duration=50.0)
#     raw.plot_psd()
    
#     plt.show()
#     plt.close()
