import os
from glob import glob
from tqdm import tqdm
from typing import Callable

import pandas as pd
import numpy as np
from prettytable import PrettyTable

from generics.configs import DataConfig, Paths, Generics
from utils.general_utils import get_logger


class CustomRawDataset():
    """
    Custom Dataset for EEG data.

    Attributes:
        config (DataConfig): Configuration object containing dataset parameters.
        augment (bool): Flag to determine if augmentation should be applied.
        mode (str): Operating mode ('train' or 'test').
        spectrograms (dict): Dictionary to hold EEG spectrograms.
        eeg_spectrograms (dict): Dictionary to hold processed EEG spectrograms.
        main_df (pd.DataFrame): Main dataframe holding the dataset information.
        label_cols (List[str]): List of label column names in the dataframe.
    """

    def __init__(
        self,
        config: DataConfig,
        subset_sample_count: int = 0,
        mode: str = "train",
        cache: bool = True,
        feature_func: Callable | None = None,
        feature_func_kwargs: dict = {}
    ):
        """
        Initialize the dataset.

        Args:
            config (DataConfig): Configuration for the dataset.
            subset_sample_count (int): Number of samples to subset. Default is 0 (all samples).
            augment (bool): Whether to apply augmentation. Default is False.
            mode (str): Operating mode ('train' or 'test'). Default is 'train'.
        """
        self.logger = get_logger("data_loader.log")
        self.config = config
        self.mode = mode
        self.main_df = pd.DataFrame()
        self.label_cols = []
        
        self.features_per_sample: np.ndarray | None = None
        self.lbl_probabilities: np.ndarray | None = None
        self.subsample_eeg_ids: np.ndarray | None = None

        cache_file = self.generate_cache_filename(subset_sample_count, mode)
        if os.path.exists(cache_file) and cache:
            self.logger.info(f"Loading dataset from cache: {cache_file}")
            self.load_from_cache(cache_file)
        else:
            self.logger.info("Processing and caching new dataset")
            self.load_data(subset_sample_count)
            
            feature_func = self.extract_features if feature_func is None else feature_func
            self.load_x_y(feature_func=feature_func, feature_func_kwargs=feature_func_kwargs)
            
            if cache:
              self.cache_data(cache_file)
    
    def load_x_y(self, feature_func: Callable, feature_func_kwargs: dict = {}):
        eeg_ids = np.asarray(self.main_df["eeg_id"])
        eeg_paths = glob(f"{Paths.TRAIN_EEGS}*.parquet")

        eegs = self.load_eegs_from_parquet(eeg_paths, eeg_ids)

        channels = [self.config.EKG_FEAT]
        [channels.extend(i) for i in self.config.FEATS]
        channels = list(set(channels))
        
        one_hot_df = self.get_one_hot(channels)
        
        expert_lbls = np.zeros((len(self.main_df), 6))
        subsample_eeg_ids = np.zeros(len(self.main_df))
        
        sample_eeg = pd.read_parquet(f"{Paths.TRAIN_EEGS}{eeg_ids[0]}.parquet").iloc[:5]
        features_per_sample = np.zeros(((len(self.main_df), len(feature_func(sample_eeg, channels, one_hot_df, **feature_func_kwargs)))))
        
        subsamples_added = 0
        for i, (eeg_id, eeg) in tqdm(enumerate(eegs)):
            eeg = self.cleanup_nans(eeg)
            if eeg_id in list(self.main_df["eeg_id"]):
                subsamples = self.main_df[self.main_df["eeg_id"] == eeg_id]
                for j, (df_i, subsample) in enumerate(subsamples.iterrows()):
                    subsample_eeg = eeg.iloc[
                        int(subsample["eeg_label_offset_seconds"] * 200) :
                        int(subsample["eeg_label_offset_seconds"] * 200 + (10 * 200))
                    ]
                    if self.mode == "train":
                        expert_lbls[subsamples_added + j -1] = np.asarray(subsample[self.label_cols])
                    subsample_eeg_ids[subsamples_added + j -1] = subsample["eeg_sub_id"]
                    features_per_sample[subsamples_added + j -1] = feature_func(subsample_eeg, channels, one_hot_df, **feature_func_kwargs)

                subsamples_added += len(subsamples)
        
        self.features_per_sample = features_per_sample # x
        self.lbl_probabilities = expert_lbls / np.sum(expert_lbls, axis=1)[:,None] # y
        self.subsample_eeg_ids = subsample_eeg_ids #TODO check if these are unique throughout main_df

    def load_eegs_from_parquet(self, eeg_paths, eeg_ids):
        for parquet_path in tqdm(eeg_paths):
            eeg_id = int(parquet_path.split("/")[-1].split(".")[0])
            if eeg_id in eeg_ids:
                yield eeg_id, pd.read_parquet(parquet_path)
                
    def get_one_hot(self, channels) -> pd.DataFrame:
        """Create one-hot encoding per EEG channel, encoding the channel group(s) they are in"""
        one_hot = {channel: [] for channel in channels}
        one_hot["EKG"] = []
        for group in self.config.FEATS:
            for channel in channels:
                one_hot[channel].append(int(channel in group))
        return pd.DataFrame(one_hot, index=self.config.NAMES)
    
    def extract_features(self, eeg_subsample: pd.DataFrame, channels, one_hot_df) -> np.ndarray:
        """Extract features from eeg subsample (mean, std, etc) 
            and combine with one-hot encoding per channel

        Args:
            eeg_subsample (pd.DataFrame): subsample from an eeg (dataframe with 2000 rows)

        Returns:
            np.ndarray: 1D numpy array with features 
        """
        desc = eeg_subsample.describe()[channels].iloc[1:]
        feature_df = pd.concat([desc, one_hot_df])
        feature_array = np.asarray(feature_df).flatten("F")

        return feature_array
    
    def cleanup_nans(self, eeg: pd.DataFrame):
        """fill NaN values with the mean of each column"""
        mean = eeg.mean()
        return eeg.fillna(mean)

    def generate_cache_filename(self, subset_sample_count: int, mode: str) -> str:
        """
        Generates a unique filename for caching the dataset based on the provided configuration.

        Args:
            subset_sample_count (int): Number of samples to subset.
            mode (str): Dataset mode ('train' or 'test').

        Returns:
            str: Filename for caching the dataset.
        """
        config_summary = f"CustomRawDataset_{subset_sample_count}_{mode}"
        return os.path.join(Paths.CACHE_PATH, f"{config_summary}.npz")

    def cache_data(self, cache_file: str):
        """
        Caches the dataset to a file.

        Args:
            cache_file (str): The file path where the dataset will be cached.
        """
        np.savez(cache_file, 
                 main_df=self.main_df.to_records(index=False),
                 features_per_sample=self.features_per_sample,
                 subsample_eeg_ids=self.subsample_eeg_ids,
                 lbl_probabilities=self.lbl_probabilities)
        self.logger.info(f"Dataset cached at {cache_file}")

    def load_from_cache(self, cache_file: str):
        """
        Loads the dataset from the cache file.

        Args:
            cache_file (str): The file path from which the dataset will be loaded.
        """
        cached_data = np.load(cache_file, allow_pickle=True)
        self.main_df = pd.DataFrame.from_records(cached_data['main_df'])
        self.features_per_sample = cached_data['features_per_sample']
        self.lbl_probabilities = cached_data['lbl_probabilities']
        self.subsample_eeg_ids = cached_data['subsample_eeg_ids']
        self.label_cols = self.main_df.columns[-6:].tolist()

    def load_data(self, subset_sample_count: int = 0):
        """
        Load data from CSV files into a DataFrame. If 'subset_sample_count' is specified, 
        it ensures unique samples based on different 'patient_id', picking one sample per patient.

        Args:
            subset_sample_count (int): Number of unique samples to load based on 'patient_id'. Default is 0 (load all samples).
        """
        try:
            csv_path = Paths.TRAIN_CSV if self.mode == "train" else Paths.TEST_CSV
            main_df = pd.read_csv(csv_path)
            main_df = main_df[~main_df["eeg_id"].isin(Generics.OPT_OUT_EEG_ID)]
            self.label_cols = main_df.columns[-6:].tolist()

            if subset_sample_count > 0:
                unique_patients = main_df['patient_id'].nunique()

                if subset_sample_count > unique_patients:
                    self.logger.warning(f"Requested {subset_sample_count} samples, but only {unique_patients} unique patients are available.")
                    subset_sample_count = unique_patients

                # Sample one record from each unique patient
                sampled_df = main_df.groupby('patient_id').sample(n=1, random_state=42).reset_index(drop=True)
                # If needed, further sample to meet the subset_sample_count
                if subset_sample_count < unique_patients:
                    sampled_df = sampled_df.sample(n=subset_sample_count, random_state=42).reset_index(drop=True)

                self.main_df = sampled_df
            else:
                self.main_df = main_df

            self.logger.info(f"{self.mode} DataFrame shape: {self.main_df.shape}")
            self.logger.info(f"Labels: {self.label_cols}")

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
  
    def print_summary(self):
          """
          Prints a summary of the dataset including dataset size, mode, data distribution,
          and vote statistics if in 'train' mode.
          """
          try:
              total_samples = len(self.main_df)
              unique_patients = self.main_df['patient_id'].nunique()
              unique_eegs = self.main_df['eeg_id'].nunique()
              unique_spectrograms = self.main_df['spectrogram_id'].nunique()

              print(f"Dataset Summary:")
              print(f"Mode: {self.mode}")
              print(f"Total Samples: {total_samples}")
              print(f"Unique Patients: {unique_patients}")
              print(f"Unique EEGs: {unique_eegs}")
              print(f"Unique Spectrograms: {unique_spectrograms}")

              if self.mode == "train":
                  label_distribution = self.main_df[self.label_cols].sum()
                  print(f"Label Distribution:\n{label_distribution}")

                  # Vote statistics
                  vote_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
                  vote_stats = self.main_df[vote_cols].agg(['mean', 'median', 'var'])
                  print("\nVote Statistics:")
                  print(vote_stats)

            #   print(f"Spectrograms Loaded: {len(self.spectrograms)}")
            #   print(f"EEG Spectrograms Loaded: {len(self.eeg_spectrograms)}")
              print(f"Probabilities Loaded: {len(self.lbl_probabilities)}")
              print(f"Features Loaded: {len(self.features_per_sample)}")
              
              
              # Configuration summary
              config_table = PrettyTable()
              config_table.field_names = ["Configuration", "Value"]
              config_table.align = "l"
              for attr in dir(self.config):
                  if not attr.startswith("__") and not callable(getattr(self.config, attr)):
                      value = getattr(self.config, attr)
                      config_table.add_row([attr, value])

              print("\nConfiguration Summary:")
              print(config_table)

          except Exception as e:
              self.logger.error(f"Error printing dataset summary: {e}")
              raise
