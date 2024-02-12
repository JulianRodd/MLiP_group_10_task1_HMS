import os
import gc
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import albumentations as A
from utils.data_preprocessing_utils import create_non_overlapping_eeg_crops
from generics import Paths, Generics
from prettytable import PrettyTable
from utils.general_utils import get_logger
from utils.loader_utils import load_eeg_spectrograms, load_spectrograms
from utils.visualisation_utils import plot_eeg_combined_graph, plot_spectrogram
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
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
        config,
        augment: bool = False,
        mode: str = "train",
        cache: bool = True,
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
        self.augment = augment
        self.mode = mode
        self.spectrograms = {}
        self.eeg_spectrograms = {}
        self.main_df = pd.DataFrame()
        self.label_cols = []
        
        if mode == "test":
            self.batch_size = config.BATCH_SIZE_TEST
        elif mode == "val":
            self.batch_size = config.BATCH_SIZE_VAL
        else:
            self.batch_size = config.BATCH_SIZE_TRAIN

        cache_file = self.generate_cache_filename(self.config.SUBSET_SAMPLE_COUNT, mode)
        if os.path.exists(cache_file) and cache:
            self.logger.info(f"Loading dataset from cache: {cache_file}")
            self.load_from_cache(cache_file)
        else:
            self.logger.info("Processing and caching new dataset")
            self.load_data(self.config.SUBSET_SAMPLE_COUNT)
            self.eeg_spectrograms = load_eeg_spectrograms(main_df=self.main_df, mode=self.mode, feats = self.config.FEATS, use_wavelet=self.config.USE_WAVELET)
            self.spectrograms = load_spectrograms(main_df=self.main_df, mode=self.mode)
            if self.mode == "train" and config.ONE_CROP_PER_PERSON:
                self.main_df = create_non_overlapping_eeg_crops(self.main_df, self.label_cols)
            if cache:
              self.cache_data(cache_file)

    def generate_cache_filename(self, subset_sample_count: int, mode: str) -> str:
        """
        Generates a unique filename for caching the dataset based on the provided configuration.

        Args:
            subset_sample_count (int): Number of samples to subset.
            mode (str): Dataset mode ('train' or 'test').

        Returns:
            str: Filename for caching the dataset.
        """
        config_summary = f"CustomDataset_{subset_sample_count}_{mode}_{self.config.ONE_CROP_PER_PERSON}"
        return os.path.join(Paths.CACHE_PATH, f"{config_summary}.npz")

    def cache_data(self, cache_file: str):
        """
        Caches the dataset to a file.

        Args:
            cache_file (str): The file path where the dataset will be cached.
        """
        np.savez(cache_file, 
                 main_df=self.main_df.to_records(index=False),
                 spectrograms=self.spectrograms,
                 eeg_spectrograms=self.eeg_spectrograms)
        self.logger.info(f"Dataset cached at {cache_file}")

    def load_from_cache(self, cache_file: str):
        """
        Loads the dataset from the cache file.

        Args:
            cache_file (str): The file path from which the dataset will be loaded.
        """
        cached_data = np.load(cache_file, allow_pickle=True)
        self.main_df = pd.DataFrame.from_records(cached_data['main_df'])
        self.spectrograms = cached_data['spectrograms'].item()
        self.eeg_spectrograms = cached_data['eeg_spectrograms'].item()
        self.label_cols = Generics.LABEL_COLS


    def load_data(self, subset_sample_count: int = 0):
        """
        Load data from CSV files into a DataFrame. If 'subset_sample_count' is specified, 
        it ensures unique samples based on different 'patient_id', picking one sample per patient.

        Args:
            subset_sample_count (int): Number of unique samples to load based on 'patient_id'. Default is 0 (load all samples).
        """
        try:
            csv_path = Paths.TEST_CSV if self.mode == "test" else Paths.TRAIN_CSV
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

                main_df = sampled_df

            if self.mode == 'val':
                _, main_df = train_test_split(main_df, test_size=self.config.VAL_SPLIT_RATIO, random_state=42)
            elif self.mode == 'train':
                main_df, _ = train_test_split(main_df, test_size=self.config.VAL_SPLIT_RATIO, random_state=42)

            self.main_df = main_df

            self.logger.info(f"{self.mode} DataFrame shape: {self.main_df.shape}")
            self.logger.info(f"Labels: {self.label_cols}")

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    
    def get_torch_data_loader(self):
        """
        Get the torch data loader for the dataset.

        Args:
            config (DataConfig): Configuration for the dataset.

        Returns:
            DataLoader: Data loader for the dataset.
        """
        try:
            return DataLoader(
                self,
                batch_size=self.batch_size,
                shuffle=self.config.SHUFFLE_TRAIN,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=self.config.PIN_MEMORY,
                drop_last=self.config.DROP_LAST,
            )
        except Exception as e:
            self.logger.error(f"Error getting data loader: {e}")
            raise
      
            

    def __len__(self) -> int:
        """
        Get the total number of items in the dataset.

        Returns:
            int: Total number of items.
        """
        return len(self.main_df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the data and its corresponding label.
        """
        try:
            X, y = self.generate_data(index)
            if self.augment:
                X = self.apply_transform(X)
            return torch.tensor(X, dtype=torch.float32), torch.tensor(
                y, dtype=torch.float32
            )
        except Exception as e:
            self.logger.error(f"Error getting item at index {index}: {e}")
            raise

    def generate_data(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a single batch of data.

        This method processes the EEG data to create input features (X) and targets (y)
        based on the specified index in the dataset.

        Args:
            index (int): Index of the batch in the dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the input features and the target values.

        Raises:
            IndexError: If the index is out of the range of the dataset.
            ValueError: If there are issues in processing the data.
        """
        try:
            if index >= len(self.main_df):
                raise IndexError(f"Index {index} is out of range for the dataset.")

            X = np.zeros((128, 256, 8), dtype="float32")
            y = np.zeros(6, dtype="float32")
            row = self.main_df.iloc[index]

            r = 0 if self.mode == "test" else int((row["min"] + row["max"]) // 4)

            for region in range(4):
                img = self.spectrograms[row.spectrogram_id][
                    r : r + 300, region * 100 : (region + 1) * 100
                ].T

                # Processing steps: Log transformation and standardization
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)
                mu = np.nanmean(img.flatten())
                std = np.nanstd(img.flatten())
                img = (img - mu) / (std + 1e-6)
                img = np.nan_to_num(img, nan=0.0)

                X[14:-14, :, region] = img[:, 22:-22] / 2.0

            if self.mode != "test":
                y = row[self.label_cols].values.astype(np.float32)

            # Append additional EEG data if available
            if row.eeg_id in self.eeg_spectrograms:
                X[:, :, 4:] = self.eeg_spectrograms[row.eeg_id]

            return X, y

        except IndexError as e:
            self.logger.error(f"Data generation error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in data generation: {e}")
            raise ValueError(f"Error processing data at index {index}")

    def apply_transform(self, img: np.ndarray) -> np.ndarray:
        """
        Apply transformations to the image.

        Args:
            img (np.ndarray): Image to transform.

        Returns:
            np.ndarray: Transformed image.
        """
        transform = A.Compose([A.HorizontalFlip(p=0.5)])
        return transform(image=img)["image"]
  
  
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
                  print(f"Augmentation: {'Enabled' if self.augment else 'Disabled'}")
                  label_distribution = self.main_df[self.label_cols].sum()
                  print(f"Label Distribution:\n{label_distribution}")

                  # Vote statistics
                  vote_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
                  vote_stats = self.main_df[vote_cols].agg(['mean', 'var'])
                  print("\nVote Statistics:")
                  print(vote_stats)

              print(f"Spectrograms Loaded: {len(self.spectrograms)}")
              print(f"EEG Spectrograms Loaded: {len(self.eeg_spectrograms)}")
              
              
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


    def plot_samples(self, n_samples: int = 2):
        """
        Plots n_samples samples each from eeg_spectrograms and spectrograms with corresponding scores.
        """
        try:
            # Plot 2 samples from eegs
            eeg_sample_ids = random.sample(list(self.eeg_spectrograms.keys()), n_samples)
            for eeg_id in eeg_sample_ids:
                plot_eeg_combined_graph(self.eeg_spectrograms[eeg_id])

            # Plot 2 samples from spectrogram_eegs
            spectrogram_sample_ids = random.sample(list(self.spectrograms.keys()), n_samples)
            for spectrogram_id in spectrogram_sample_ids:
                plot_spectrogram(self.spectrograms[spectrogram_id])

        except Exception as e:
            self.logger.error(f"Error plotting samples: {e}")
            raise
