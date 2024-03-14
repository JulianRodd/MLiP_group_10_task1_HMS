import librosa
import mne
import numpy as np
import pandas as pd


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

        validate_config(custom_config)

        ch_types = ["eeg"] * (len(eeg.columns) - 1)
        ch_types.append("ecg")

        with mne.use_log_level("warning"):
            info = mne.create_info(
                ch_names=eeg.columns.tolist(),
                sfreq=custom_config["sfreq"],
                ch_types=ch_types,
            )

            raw: mne.io.RawArray = mne.io.RawArray(eeg.values.T, info)

            montage = mne.channels.make_standard_montage("standard_1020")
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

        img = np.zeros((128, 256, 4), dtype="float32")
        signals = []

        for k in range(4):
            COLS = feats[k]

            for kk in range(4):
                x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values
                m = np.nanmean(x)
                x = np.nan_to_num(x, nan=m)

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

        return img
    except Exception as e:
        print(f"Error processing EEG data: {e}")
        raise


def validate_config(config):
    if "sfreq" not in config:
        raise ValueError(
            f"eeg sample frequency ('sfreq') not in custom_preprocessing_config."
        )
    return
